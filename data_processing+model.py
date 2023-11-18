import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Dense
import tensorflow as tf
import pandas as pd

""" 
Function that loads all the image paths from the datastet.csv file and uses them to access all the images
in the hard drive and organizes them in to an images list and a labels list. Each image goes through 12 iterations
of random image augmentation functions that alter the brightness and orientation of the image.
"""
def load_images_from_folder(target_size=(224, 224)):

    def rand_rotate(image, maxangle=90):
        maxangle = np.random.randint(-maxangle, maxangle)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, maxangle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated_image


    def rand_flip(image, probability=0.5):
        if np.random.rand() < probability:
            flipped_image = cv2.flip(image, 1) 
            return flipped_image
        else:
            return image
        
    def rand_brightness(image, brightness=(0.5, 1.5)):
        brightness_factor = np.random.uniform(brightness[0], brightness[1])
        adjusted_image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)
        return adjusted_image
    
    def image_augmentation(image):

        image = rand_brightness(image)
        image = rand_flip(image)
        image = rand_rotate(image)
        return image

    csv_file = 'your path'
    data = pd.read_csv(csv_file)

    image_paths = data['image_path'].tolist()
    labels1 = data['label'].tolist()


    images = []
    labels = []

    for image_path, label in zip(image_paths, labels1):
        image = cv2.imread(image_path)
        if image is not None:
            for x in range(12):
                image2 = image_augmentation(image)
                image3 = cv2.resize(image2, target_size)
                images.append(image3)
                labels.append(label)


    images = np.array(images)
    labels = np.array(labels)

    return images, labels

#dataset = '/Volumes/Lexar/dataset'
X, y = load_images_from_folder()

# Label Encoding
np.set_printoptions(threshold=np.inf)


class_to_int = {'flat_head_screws': 0, 'pan_head_screws': 1, 'hex_bolts': 2, 'slip_joint_pliers': 3, 'adjustable_wrench': 4, 'hex_key': 5, 'combination_wrench': 6,
                'c_clamp': 7, 'funnel': 8, 'level': 9, 'siff': 10, 'spachula': 11, 'tape': 12, 'tape_measure': 13}

# Encoding labels
intlabels = []
for x in y:
    intlabels.append(class_to_int[x])

intlabels = np.array(intlabels)
num_classes = len(class_to_int)
one_hot_encoded_labels = np.eye(num_classes)[intlabels]


# Data Normalization
X = np.array(X, dtype='float32') / 255.0

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, one_hot_encoded_labels, test_size=0.2, random_state=42)

# Randomize the order of the training data
num_samples = X_train.shape[0]
random_indices = np.random.permutation(num_samples)
X_train_shuffled = X_train[random_indices]
y_train_shuffled = y_train[random_indices]


model = tf.keras.Sequential()


# Add your custom layers on top of the base model
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add((MaxPooling2D(pool_size=(2, 2))))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(14, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Training
model.fit(X_train_shuffled, y_train_shuffled, epochs=20, batch_size=22, validation_split=0.1) 

# Model Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

#Saving Model
model.save('your path')
 

