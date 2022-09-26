# -*- coding: utf-8 -*-
import warnings
import os

from CK.CK_process import ck_process

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import load_model
from load_and_process import load_fer2013, preprocess_input, load_fer2013plus
from sklearn.model_selection import train_test_split

"""
Function: Import the trained model, traverse the test set for prediction, calculate the confusion matrix and draw the output, save
File location：\Emotion Recognition\plot_confusion_matrix.py

"""

# Emoji Category Tags
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
#fer2013
EMOTIONSp = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear","contempt"]
#fer2013+
EMOTIONSc = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
#ck+
# Model location
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
emotion_model_pathplus = 'models/PLUS_mini_XCEPTION.183-0.8064.hdf5'
emotion_model_pathck = 'models/ck_mini_XCEPTION.122-0.9396.hdf5'

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary,EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(EMOTIONS)))
    plt.xticks(xlocations, EMOTIONS, rotation=45)
    plt.yticks(xlocations, EMOTIONS)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# load datasets
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
facesp, emotionsp = load_fer2013plus()
facesp = preprocess_input(facesp)
facesc, emotionsc = ck_process()
facesc = preprocess_input(facesc)

def Output_matrix(emotion_model_path,title,EMOTIONS,faces,emotions):
    emotion_model = load_model(emotion_model_path, compile=False)  # load model
    input_shape = emotion_model.input_shape[1:]  # the input size of the model
    # Load dataset
    #faces, emotions = load_fer2013()
    #faces, emotions = load_fer2013plus()
    #faces, emotions = ck_process()
    #faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape

    # Split training and test sets
    xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)

    # Use the trained model to make predictions on the test set
    ndata = xtest.shape[0]  # Test set data volume
    y_pred = np.zeros((ndata,))  # Used to save predicted results
    y_true = [ytest[i].argmax() for i in range(ndata)]  # Get real labels
    y_true = np.array(y_true)  # matrix

    # Traverse the test set to get the prediction results
    for i in range(ndata):
        input_image = xtest[i]
        input_image = cv2.resize(input_image, input_shape[0:2], cv2.INTER_NEAREST)
        # Make sure the input dimensions match the model input
        input_image = np.reshape(input_image, (1, input_shape[0], input_shape[1], input_shape[2]))
        # Invoke the model to make predictions for each image
        preds = emotion_model.predict(input_image)[0]
        y_pred[i] = preds.argmax()  # The maximum position is the final result

    tick_marks = np.array(range(len(EMOTIONS))) + 0.5

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=4)  # set precision
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # ratio of these classifications
    print('confusion matrix：')
    print(cm_normalized)
    accuracy = np.mean([cm_normalized[i, i] for i in range(num_classes)])  # The mean of the sum of the results on the right slash of the confusion matrix is the accuracy
    print('Accuracy：' + str(round(accuracy, 4)))

    # Create a window to plot the confusion matrix
    plt.figure(figsize=(12, 8), dpi=120)
    ind_array = np.arange(len(EMOTIONS))
    x, y = np.meshgrid(ind_array, ind_array)

    # Add per-bin classification ratio results
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.4f" % (c,), color='red', fontsize=10, va='center', ha='center')
    # Set up the chart
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(False, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show drawing
    plot_confusion_matrix(cm_normalized, title=title,EMOTIONS = EMOTIONS)
    plt.savefig('confusion_matrix.png', format='png')  # save result
    plt.show()  # display window
Output_matrix(emotion_model_path,"FER2013 Confusion Matrix",EMOTIONS,faces, emotions)
Output_matrix(emotion_model_pathplus,"FER2013plus Confusion Matrix",EMOTIONSp,facesp, emotionsp)
Output_matrix(emotion_model_pathck,"CK+ Confusion Matrix",EMOTIONSc,facesc, emotionsc)