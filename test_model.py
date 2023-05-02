import os
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.applications import ResNet50

label_map = {'bmw serie 1': 0,
'chevrolet spark': 1,
'chevroulet aveo': 2,
'clio': 3,
'duster': 4,
'golf': 5,
'hyundai i10': 6,
'hyundai tucson': 7,
'logan': 8,
'megane': 9,
'mercedes class a': 10,
'nemo citroen': 11,
'octavia': 12,
'picanto': 13,
'polo': 14,
'sandero': 15,
'seat ibiza': 16,
'symbol': 17,
'toyota corolla': 18,
'volkswagen tiguan': 19}

# Define the command line arguments
parser = argparse.ArgumentParser(description='Testing a pre-trained TensorFlow model on a folder of images')
parser.add_argument('--model_path', type=str, help='path to the pre-trained TensorFlow model')
parser.add_argument('--image_folder_path', type=str, help='path to the folder with images to be tested')
parser.add_argument('--output_file_path', type=str, help='path to the output text file')
args = parser.parse_args()
img_shape = (224, 224, 3)
# Load the model from the .h5 file
model = tf.keras.models.load_model(args.model_path)

# Define the preprocessing function for the test images
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=img_shape[:2])
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

# Set the directory containing the test images
test_dir = args.image_folder_path

# Create a dictionary to hold the results
results = {}

# Iterate through the test images and make predictions using the model
for filename in os.listdir(test_dir):
    if filename.endswith('.jpg'):
        # Load and preprocess the image
        img_path = os.path.join(test_dir, filename)
        img = preprocess_image(img_path)

        # Make a prediction using the model
        prediction = model.predict(img)
        label = np.argmax(prediction)

        # Store the prediction result in the dictionary
        label_name = list(label_map.keys())[list(label_map.values()).index(label)]
        if label_name in results:
            results[label_name].append(filename)
        else:
            results[label_name] = [filename]

# Print the results
for label, filenames in results.items():
    count = len(filenames)
    print(f"{label}: {count} - {filenames}")

# Save the results to a text file
with open(args.output_file_path, 'w') as f:
    for label, filenames in results.items():
        count = len(filenames)
        f.write(f"{label}: {count} - {str(filenames)}\n")
