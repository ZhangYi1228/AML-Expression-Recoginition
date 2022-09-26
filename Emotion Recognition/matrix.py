# for displaying the confusion matrix, generate a csv file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
start_time = time.time()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from keras_preprocessing.image import ImageDataGenerator
from keras.models import load_model
datagen=ImageDataGenerator(rescale=1.0/255)
model=load_model('./models/_mini_XCEPTION.102-0.66.hdf5')
test=datagen.flow_from_directory('D:/Datasets/datasets/test',
                                target_size=(48,48),
                                batch_size=8,
                                class_mode='categorical',
                                shuffle=False)
from keras.utils import np_utils
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
test_labels=np_utils.to_categorical(test.classes)
y_true=test_labels.argmax(axis=1)
y_pred=model.predict_generator(test,len(test.classes)/test.batch_size)
y_pred=y_pred.argmax(axis=1)
print(y_true.shape,y_pred.shape)
#print(y_true,y_pred)
uniques=np.unique(y_true,axis=0)
print(y_true.shape,y_pred.shape)
classify_report=metrics.classification_report(y_true,y_pred)
confusion_matrix=metrics.confusion_matrix(y_true,y_pred)
overall_accuracy=metrics.accuracy_score(y_true,y_pred)
acc_for_each_class=metrics.precision_score(y_true,y_pred,average=None)
average_accuracy=np.mean(acc_for_each_class)
score=metrics.accuracy_score(y_true,y_pred)
print('classify_report:\n',classify_report)
print('confusion_matrix:\n',confusion_matrix)
import pandas as  pd
data1=pd.DataFrame(confusion_matrix)
data1.to_csv('confusion_matrix.csv')
 
print('acc_for_each_class:\n',acc_for_each_class)
print('average_accuracy:{0:f}'.format(average_accuracy))
print('overall_accuracy:{0:f}'.format(overall_accuracy))
print('score:{0:f}'.format(score))
