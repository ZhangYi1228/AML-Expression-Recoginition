
import cv2
import imutils
import numpy as np
from PyQt5 import QtGui, QtWidgets
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from load_and_process import preprocess_input


class Emotion_Rec:
    def __init__(self, model_path=None):

        # Parameters for loading data and images
        detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'

        if model_path == None:  # If no path is specified, the default model is used
            emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
        else:
            emotion_model_path = model_path

        # Load face detection model
        self.face_detection = cv2.CascadeClassifier(detection_model_path)  # Cascading Classifiers

        # Cascading Classifiers
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        # Expression category

        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "contempt"]
        #self.EMOTIONS = ["neutral","happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
        #self.EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

    def run(self, frame_in, canvas, label_face, label_result):
        # frame_in Camera screen or image
        # canvas background image for display
        # label_face The label object used for the face display
        # label_result the label object used to display the results

        # Adjust the screen size
        frame = imutils.resize(frame_in, width=300)  # zoom screen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Detect faces
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                                     minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        preds = []  # predicted outcome
        label = None  # predicted label
        (fX, fY, fW, fH) = None, None, None, None  # face position
        frameClone = frame.copy()  # Copy screen

        if len(faces) > 0:
            # Sort detected faces according to ROI size
            faces = sorted(faces, reverse=False, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))  # 按面积从小到大排序

            for i in range(len(faces)):  # Traverse each detected face, identify all faces by default
                # If you only want to recognize and display the largest face, you can uncomment the if...else code snippet here
                # if i == 0:
                #     i = -1
                # else:
                #     break

                (fX, fY, fW, fH) = faces[i]

                # Extract a region of interest (ROI) from the grayscale image, convert its size to the same dimensions as the model input, and prepare the ROI for the classifier passing through the CNN
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, self.emotion_classifier.input_shape[1:3])
                roi = preprocess_input(roi)
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                '''
                lbp = LBP()
                image_array = lbp.describe(roi)
                roi = lbp.lbp_basic(image_array)
                '''

                # Use the model to predict the probability of each class
                preds = self.emotion_classifier.predict(roi)[0]
                # emotion_probability = np.max(preds)  # maximum probability
                label = self.EMOTIONS[preds.argmax()]  # Select the expression class with the highest probability

                # Circle the face area and display the recognition result
                cv2.putText(frameClone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0, 255, 0), 1)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (255, 255, 0), 1)

        #canvas = 255* np.ones((250, 300, 3), dtype="uint8")
        #canvas = cv2.imread('slice.png', flags=cv2.IMREAD_UNCHANGED)

        for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):
            # Used to display the probability of each class
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # Plot a bar chart of expression classes and corresponding probabilities
            w = int(prob * 300) + 7
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (224, 200, 130), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

        # Adjust the screen size to fit the interface
        frameClone = cv2.resize(frameClone, (420, 280))

        # Display face in Qt interface
        show = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
        QtWidgets.QApplication.processEvents()

        # Display the result in the label where the result is displayed
        show = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_result.setPixmap(QtGui.QPixmap.fromImage(showImage))

        return (label)
