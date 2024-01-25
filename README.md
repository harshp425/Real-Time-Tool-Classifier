# Real-Time-Tool-Classifier

This is a real time object detection program which leverages computer vision and machine learning to classify different tools. The model is trained on a custom dataset consisting of over 5500 images featuring a variety of screws, nuts, and tools with an accuracy rate of over 80%. The model uses OpenCV to access the webcam and provide real-time classification of equipment with additional instructions on use and return location. The program was implemented in Cornell’s Bovay Civil Infrastructure Lab to improve organization within the complex and assist students/researchers in safely and efficiently utilizing tools.

## Technologies Used

This project is written in Python leveraging cv2 for image alteration and augmentation which allow the dataset that the model is trained on to become much more diversified in terms of the image size, scale, intensity, brightness, colors, etc. It is also used in the `webcam_real_time.py` file as well in order to access the webcam for real time feed. NumPy is primarily used to ensure the randomization of the augmentations we apply to each image in the dataset and to organize the the classes or images along with their labels. Additionally, sklearn, keras, and TensorFlow are used in to provide the essential building, tranining, and testing methods for the model. 

## Key Features

The functionality that this model provides to users includes...
- Real-time detection and classification of...
    - Flat-headed Screws
    - Pan-Headed Screws
    - Hex Nuts
    - Slip Joint Plier
    - Adjustable Wrench
    - Hex Key
    - Combination Wrench
    - C-Clamps
    - Funnels
    - Level Bars
    - Siffs
    - Spachulas
    - Tape
    - Tape-Measure
- Information on where each of the items are located (the program was created for use in a specific area and so location details are curated towards that area; however, location details can be altered)

## Installation and Setup

In order to run the project, it is reccomended that the user has a text editor such as VsCode installed but they can also run the program through the terminal as well. The user should create a project folder and copy the files/folders in the GitHub repository. Once downloaded, the folder/file setup should be similar to the example below...



