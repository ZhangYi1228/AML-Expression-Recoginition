# for data augmentation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

#dir is the folder where you want to perform data enhancement, save_to_dir is the location where the enhanced images are stored
dir = 'D:/Datasets/datasets/train/5'

for filename in os.listdir(dir):  # The argument to listdir is the path to the folder
    print(filename)
    img = load_img(dir + '/' + filename)  # Here is a PIL image
    x = img_to_array(img)  # Convert the PIL image to a numpy array of shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # This is a numpy array of shape (1, 3, 150, 150)

    # All images produced are stored in the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='D:/Datasets/datasets/fer2013TrainDataAdd/5',
                              save_prefix='5',
                              save_format='jpeg'):
        i += 1
        if i > 5:
            break  # Otherwise the generator will exit the loop