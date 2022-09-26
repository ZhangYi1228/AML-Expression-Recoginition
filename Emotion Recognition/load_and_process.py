# for loading and process the datasets
import pandas as pd
import cv2
import numpy as np

from LBP.LBP_ALL import rotation_invariant_LBP
from LBP.get_LBP_from_Image import LBP

dataset_path = 'C:/Users/zy/Desktop/Emotion Recognition/fer2013/fer2013/fer2013.csv'# 文件保存位置
datasetplus_path = 'C:/Users/zy/Desktop/Emotion Recognition/fer2013/fer2013/fer2013plus.csv'

image_size=(48,48)# image size
#image_size1=(120,120)# image size
lbp=LBP()

# load dataset
def load_fer2013():
        data = pd.read_csv(dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            #print(face)
            face = lbp.lbp_revolve(face)
            #print(face)
            face = cv2.resize(face.astype('uint8'),image_size)
            #print(face)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        #print(faces[0])
        #print(emotions)
        return faces, emotions

def load_fer2013plus():
    data = pd.read_csv(datasetplus_path)
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        #face = lbp.lbp_basic(face)
        face = lbp.lbp_revolve(face)
        #face1 = rotation_invariant_LBP(face)
        #face = (face+face1)/2
        face = cv2.resize(face.astype('uint8'), image_size)

        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    emotions = pd.get_dummies(data['emotion']).as_matrix()
    return faces, emotions

# normalize the data
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


#load_fer2013()
