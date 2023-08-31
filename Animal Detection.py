pip install tensorflow

#importing all the modules required...
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# Add custom classification layers on top of VGG16
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)  # Replace 'num_classes' with the actual number of classes

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Load the image and resize it to the required size
img = image.load_img('image.jpg', target_size=(224, 224))

# Convert the image to a numpy array and preprocess it
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Extract features from the image using the VGG16 base model
features = base_model.predict(x)

# Use the extracted features to detect animals in the image using your custom classification model
animal_predictions = model.predict(features)

# Print the predicted probabilities for each class
print(animal_predictions)
