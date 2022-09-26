#Count the number distribution of various expressions in various data sets
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from CK.CK_process import ck_process
from load_and_process import load_fer2013, load_fer2013plus

fer = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral","contempt"]
ferplus = ["neutral","happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
ck = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise","neutral"]

def choose(num,label):
    label[num] = label[num]+1

def emotion_value(emotionlabel,emotionsset,graphname):
    labelnum = []
    for i in range(len(emotionlabel)):
        labelnum.append(0)
    #faces = preprocess_input(faces)
    for i in emotionsset:
        #print(np.argmax(i))
        choose(np.argmax(i),labelnum)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.bar(emotionlabel, labelnum, label=graphname)
    # params
    # x: bar chart x-axis
    # y：the height of the bar
    # width：The width of the bar chart is 0.8 by default
    # bottom：The y-coordinate value of the bottom of the bar is 0 by default
    # align：center / edge Whether the bar graph is centered on the x-axis coordinate or edged on the x-axis coordinate
    plt.legend()
    plt.xlabel('emotion')
    plt.ylabel('number')
    plt.title(u'Number of each emotion')

    for i in range(len(emotionlabel)):
        plt.text(i-0.25, labelnum[i] + 0.02*max(labelnum), "%s" % labelnum[i], va='center')

    plt.show()
    return labelnum

def convert(emotions,labelnum):
    combinelabel = []
    for i in range(8):
        combinelabel.append(0)
    for i in range(8):
        #"neutral", "happiness", "surprise", "sadness", "anger",\
        #"disgust", "fear", "contempt"
        a = np.array(emotions)
        b = np.array(ferplus)
        #np.where(a == b[i])
        combinelabel[i] = labelnum[emotions.index(ferplus[i])]
    return combinelabel

#
def Combine(ferset,ferplusset,ckset):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # Enter statistics
    emotions = ("neutral","happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt")
    fernum = convert(fer, emotion_value(fer, ferset,"fer2013"))
    ferplusnum = emotion_value(ferplus, ferplusset,"fer2013plus")
    cknum = convert(ck, emotion_value(ck, ckset,"CK+"))

    bar_width = 0.3  # bar width
    index_fer = np.arange(len(emotions))  # abscissa of fer bar chart
    index_ferplus = index_fer + bar_width  # abscissa of ferplus bar chart
    index_ck = index_ferplus + bar_width   #ck

    # Use the bar function twice to draw two sets of bars
    plt.bar(index_fer, height=fernum, width=bar_width, color='b', label='fer2013')
    plt.bar(index_ferplus, height=ferplusnum, width=bar_width, color='g', label='ferplus2013')
    plt.bar(index_ck, height=cknum, width=bar_width, color='r', label='CK+')


    plt.legend()  # show legend
    plt.xticks(index_fer + bar_width / 2, emotions)  # Let the abscissa axis scale display the expression classification, index_male + bar_width/2 is the position of the abscissa axis scale
    plt.ylabel('Emotion number')  # vertical axis title
    plt.title('Number of different datasets')  # figure title

    plt.show()


#fer2013
faces,ferset = load_fer2013()
#emotion_value(fer,ferset,"fer2013")
#fer2013plus
faces,ferplusset = load_fer2013plus()
#emotion_value(ferplus,ferplusset,"fer2013plus")
#ck
faces,ckset = ck_process()
#emotion_value(ck,ckset,"CK+")

Combine(ferset,ferplusset,ckset)
