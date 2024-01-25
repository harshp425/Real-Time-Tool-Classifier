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

In order to run the project, it is reccomended that the user has an IDE such as VsCode installed but they can also run the program through the terminal as well. The user should create a project folder and copy the files/folders in the GitHub repository. Once downloaded, the folder/file setup should be similar to the example below...

<img width="224" alt="Screenshot 2024-01-24 at 7 41 13 PM" src="https://github.com/harshp425/Real-Time-Tool-Classifier/assets/126726290/24818b54-cf79-4110-aefc-bfb3b46fed86">

Inside the `webcam_real_time.py` file, the user must navigate to `model = tf.keras.models.load_model('your path')` and replace the `'your path'` with the actaul path of the project folder.


## Running the Program

In order to run the program, the user must run the `webcam_real_time.py` file either through the integrated terminal of VsCode or through the terminal/shell of the machine (ensuring that the user is in the proper directory). Upon initialization of the program, a window will appear on the user's screen displaying the computer's camera view (Note: if the user is using a MacBook, the camera feed might originate from thier phone).



In order to exit the program, the user can simply press the `q` key which will terminate the program. 

## Customization 

In order to customize the location of the tools, the user must navigate to the  `webcam_real_time.py` where they will find the `location` function at the top of the file. This function creates and returns a dictionary which has all of the tools as keys and thier locations as values. The user can then just alter strings that provide the location for each of the tools. 

