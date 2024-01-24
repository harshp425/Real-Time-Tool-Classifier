# Real-Time-Tool-Classifier

This is a real time object detection program which leverages computer vision and machine learning to classify different tools. The model is trained on a custom dataset consisting of over 5500 images featuring a variety of screws, nuts, and tools with an accuracy rate of over 80%. The model uses OpenCV to access the webcam and provide real-time classification of equipment with additional instructions on use and return location. The program was implemented in Cornell’s Bovay Civil Infrastructure Lab to improve organization within the complex and assist students/researchers in safely and efficiently utilizing tools.

## Technologies Used

This project is written in Python leveraging cv2 for image alteration and augmentation which allow the dataset that the model is trained on to become much more diversified in terms of the image size, scale, intensity, brightness, colors, etc. NumPy is primarily used to ensure the randomization of the augmentations we apply to each image in the dataset and to organize the the classes or images along with their labels. Additionally, sklearn, keras, and TensorFlow are used in to provide the essential building, tranining, and testing methods for the model.


