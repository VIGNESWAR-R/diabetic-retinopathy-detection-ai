import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load labels
labels = pd.read_csv('dataset/train.csv')
labels['id_code'] = labels['id_code'].apply(lambda x: f'{x}.png')
labels['diagnosis'] = labels['diagnosis'].astype(str)

# Preprocess images
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize
    return img

# Prepare dataset
X = []
y = []

for index, row in labels.iterrows():
    image_path = os.path.join('dataset/train_images', row['id_code'])
    if os.path.exists(image_path):
        img = preprocess_image(image_path)
        X.append(img)
        y.append(row['diagnosis'])

X = np.array(X)
y = np.array(y)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)

print("Preprocessing complete! Preprocessed data saved.")