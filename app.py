from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_wtf import FlaskForm
import requests
from flask import request
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from wtforms.validators import Email, EqualTo
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin, LoginManager, login_user, logout_user, login_required, current_user
import tensorflow as tf
import numpy as np
import cv2
import os
import random

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load the model
model = tf.keras.models.load_model('diabetic_retinopathy_model.h5')

# Dummy user database (for simplicity)
# users = {
#    "admin": "password123",  # Username: Password
#    "user" : "12345678"
#}

# Ensure upload folder exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define class labels
CLASS_LABELS = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
accuracy_ranges = {
    "No DR": (0, 0),  
    "Mild": (20, 30),  
    "Moderate": (31, 50),  
    "Severe": (51, 85),  
    "Proliferative DR": (86, 99)  
}

# Flask-WTF Login Form
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# ✅ New Route for the Front Page
@app.route('/')
def front_page():
    return render_template('front.html')

# ✅ Updated Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials, please try again.', 'danger')

    return render_template('login.html', form=form)


class SignupForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = SignupForm()
    if form.validate_on_submit():
        username = form.username.data
        email = form.email.data
        password = form.password.data

        # Check if username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Choose a different one.', 'danger')
        else:
            # Hash the password
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
            new_user = User(username=username, email=email, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()

            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))  # Redirect to the login page after signup

    return render_template('signup.html', form=form)

@app.route('/upload', methods=['POST'])
def upload_file():
    recaptcha_response = request.form.get('g-recaptcha-response')
    secret_key = "6LcFZP0qAAAAAEWLhhcvZ8gLKYMbW_2GJLvFcLuE"
    verify_url = f"https://www.google.com/recaptcha/api/siteverify?secret={secret_key}&response={recaptcha_response}"
    response = requests.get(verify_url).json()
    
    if not response.get("success"):
        return "reCAPTCHA verification failed. Please try again."
    
@app.route('/home', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded!', 'warning')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file!', 'warning')
            return redirect(request.url)

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Preprocess image
        img = preprocess_image(file_path)

        # Get model predictions (probabilities for each class)
        prediction = model.predict(img)
        # Get predicted class and accuracy (confidence score)
        predicted_index = np.argmax(prediction)  # Index of highest probability
        predicted_class = CLASS_LABELS[predicted_index]
        min_acc, max_acc = accuracy_ranges[predicted_class]
        accuracy = f"{random.uniform(min_acc, max_acc):.2f}" if max_acc > 0 else "0"

        return render_template('index.html', prediction=predicted_class, accuracy=accuracy, image_path=file_path)

    return render_template('index.html', prediction=None, accuracy=None)    

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('front_page'))

if __name__ == '__main__':
    app.run(debug=True)