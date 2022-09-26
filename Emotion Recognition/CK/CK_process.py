import os
import csv
import cv2
import numpy as np
from PIL import Image
import pandas as pd

from LBP.LBP_ALL import rotation_invariant_LBP

label_dir = r'D:\CK\Emotion_labels'
image_dir = r'D:\CK\cohn-kanade-images'

# desired image size
purpose_size = 120
face_cascade = cv2.CascadeClassifier('D:/CK/haarcascade_frontalface_default.xml')
image_size=(120,120)

# Crop the face part
def image_cut(file_name):
    # cv2 reads the picture
    im = cv2.imread(file_name)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2 detects the central area of the face
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5)
    )
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # PIL reads pictures
            img = Image.open(file_name)
            # Convert to grayscale image
            img = img.convert("L")
            # Crop the core part of the face
            crop = img.crop((x, y, x + w, y + h))
            # Reduced to 120*120
            crop = crop.resize((purpose_size, purpose_size))
            return crop
    return None


# Convert image to data matrix
def image_to_matrix(filename):
    # Crop and shrink
    img = image_cut(filename)
    data = img.getdata()
    data = np.array(data, dtype=float)
    #data = np.reshape(data, (120, 120))
    #print(len(data))
    #print(data)
    return data

# Get the label value in the file
def get_label(file_name):
    f = open(file_name, 'r+')
    line = f.readline()  # only one row
    line_data = line.split(' ')
    label = float(line_data[3])
    f.close()
    # 1-7 label values are converted to 0-6
    return int(label) - 1


# Save the data of the core area of the face to a csv file
def ck_process():
    # like [[data1, label1], [data2, label2], ...]
    label_data = []
    emotion = []
    faces = []

    # Get a list of subdirectories, like ['S005\001', 'S010\001', 'S010\002', ...]
    dir_list = []
    for root, dirs, _ in os.walk(image_dir):
        for rdir in dirs:
            for _, sub_dirs, _ in os.walk(root + '\\' + rdir):
                for sub_dir in sub_dirs:
                    dir_list.append(rdir + '\\' + sub_dir)
                break
        break

    # Traverse directories to get files
    for path in dir_list:
        # process images
        for root, _, files in os.walk(image_dir + '\\' + path):
            for i in range(0, len(files)):
                if files[i].split('.')[1] == 'png':
                    # 裁剪图片，并将其转为数据矩阵
                    img_data = image_to_matrix(root + '\\' + files[i])
                    # 处理相应的 labelCrop the image and turn it into a data matrix
                    for lroot, _, lfiles in os.walk(label_dir + '\\' + path):
                        if len(lfiles) > 0:  # picture has label
                            label = get_label(lroot + '\\' + lfiles[0])
                            emotion.append(label)
                            label_data.append([label,img_data])
                            #print([label,img_data])
                        break
            break
    #print(len(label_data))
    #print(faces)
    #print(label_data)
    #print(len(faces))
    for i in label_data:
        face = np.asarray(i[1]).reshape(120 , 120)
        face = cv2.resize(face.astype('uint8'), image_size)

        faces.append(face.astype('float32'))
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    #print(len(faces))
    #print(faces)
    #print(emotion)
    emotions = pd.get_dummies(emotion).as_matrix()
    #print(emotions[15])

    return faces,emotions



if __name__ == '__main__':
    #ck_process()
    filename = 'D:/CK/cohn-kanade-images/S022/003/S022_003_00000001.png'
    #print(image_to_matrix(filename))
    cv2.imshow('Image',np.array(image_to_matrix(filename)))
    cv2.waitKey(0)
    print('\n--------------------------Program Finished---------------------------\n')