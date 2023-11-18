import cv2
import tensorflow as tf
import numpy as np
import cv2
import numpy as np

#Function to return Location 
def location(predicted_class_label):
    dict = {'Flat-headed Screw': 'Long wooden worktable by Kimball Hall entrance.', 
            'Pan-Headed Screw': 'Long wooden worktable by Kimball Hall entrance.',
            'Hex Nut': 'Long wooden worktable by Kimball Hall entrance OR in blue shelves behind the actuators.', 
            'Slip Joint Plier': 'Very bottom drawer of drawer stand to the left of the bandsaw.', 
            'Adjustable Wrench': 'Second to last drawer of drawer stand to the left of the bandsaw.',
            'Hex Key': 'Third drawer from the top of drawer stand to the left of the bandsaw. ',
            'Combination Wrench' : 'Fourth drawer from the top of drawer stand to the left of the bandsaw. ',
            'C-Clamp': 'On the board behind the bandsaw by rear exit of classroom.',
            'Funnel': 'White plastic box on the first self under the concrete lab table.',
            'Level': 'On the drawer stand next to the bandsaw.',
            'Siff': "White plastic box on the first self under the concrete lab table.",
            'Spachula': "White plastic box on the first self under the concrete lab table.",
            'Tape': "Inside the red cabinet labeled tape/welding gear by the welding station.",
            'Tape-Measure': "Very bottom drawer of drawer stand to the left of the bandsaw."
            }
    return dict[predicted_class_label]

#Function to return uses of equipment
def uses(predicted_class_label):
    dict = {'Flat-headed Screw': 'Used mainly to fasten wooden workpieces and are ideal for areas that accumulate lots of dust', 
            'Pan-Headed Screw': 'Used mainly to fasten wooden workpieces',
            'Hex Nut': 'Used with anchor shackles and bolts to connect both metal and wood components to prevent tension and movement.', 
            'Slip Joint Plier': 'Used to grip and pull materials of various thicknesses.Some also have wire cutters.',
            'Adjustable Wrench': 'Used to turn or loosen a nut or bolt AND can adjust width of jaws accordingly.',
            'Hex Key': 'Used to tighten and loosen hexagonal bolts. ',
            'Combination Wrench' : 'The closed side is for hexagonal or square nuts while the open side is used for all types of nuts.',
            'C-Clamp': 'Device used to fasten objects together to keep the conjoined or in place.',
            'Funnel': "Used to more precisely measure out the right amount of aggregate for concrete mixtures.",
            'Level': "Used to ensure that surfaces are either perfectly vertical or horizontal",
            'Siff': "Used for the fine aggregate that is added to concrete mixtures.",
            'Spachula': "Primarily used to mix concerete mixtures before test mold casting.",
            'Tape': "Used to attach items together.",
            'Tape-Measure': "Used for measuring purposes."
            }
    return dict[predicted_class_label]

model = tf.keras.models.load_model('your path')

# Access the webcam
cam_capture = cv2.VideoCapture(0)  # 0 represents the default camera (you can change it if needed)

while True:
    ret, frame = cam_capture.read()

    # Preprocess the frame (resize and normalize)
    if frame is not None and not frame.size == 0:
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
    else:
        print("Invalid frame. Unable to resize.")

    input_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension

    # Perform classification
    predictions = model.predict(input_frame)
    probabilities = np.array(predictions)
    maxindex = np.argmax(probabilities)

    # Get class label (assuming you have a list of class labels)
    class_labels = ["Flat-headed Screw", "Pan-Headed Screw", "Hex Nut", "Slip Joint Plier", "Adjustable Wrench", "Hex Key", "Combination Wrench", 'C-Clamp', 'Funnel', 'Level', 'Siff', 'Spachula', 'Tape', 'Tape-Measure']
    predicted_class_label = class_labels[maxindex]

    # Display the result on the frame
    cv2.putText(frame, predicted_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, 'Location: ' + location(predicted_class_label), (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, 'Uses: ' + uses(predicted_class_label), (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Webcam Classification', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cam_capture.release()
cv2.destroyAllWindows()
