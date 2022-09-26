"""
Description: Train a facial expression recognition program for dataset FER2013
"""
import warnings
import os

from LBP.LBP_ALL import rotation_invariant_LBP
from save_loss_acc import LossHistory

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings(action='ignore')
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import load_fer2013
from load_and_process import preprocess_input
from models.cnn import mini_XCEPTION
from sklearn.model_selection import train_test_split

# parameters
batch_size = 32
num_epochs = 10000
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = 'models/'



# construct model
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', # optimizer is adam
              loss='categorical_crossentropy', # Logarithmic Loss Function for Multi-Classification
              metrics=['accuracy'])
model.summary()




# Define the callback function Callbacks for the training process

log_file_path = base_path + 'lbp_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4),
                              verbose=1)
history = LossHistory()
# Model location and naming
trained_models_path = base_path + 'LBP_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.4f}.hdf5'

# Define model weight location, naming, etc.
model_checkpoint = ModelCheckpoint(model_names,
                                   'val_loss', verbose=1,
                                    save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr,history]



# load dataset
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape

# Divide training and test sets
xtrain, xtest,ytrain,ytest = train_test_split(faces,
                                              emotions,
                                              test_size=0.2,
                                              random_state=20,
                                              stratify = emotions,
                                              shuffle=True)



#Image generator, enhance the data in batches, expand the size of the data set
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.1,
                        horizontal_flip=True
                        )

'''
data_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

'''

# Train with data augmentation
model.fit_generator(data_generator.flow(xtrain, ytrain, batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs,
                        verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest))
history.acc_loss_plot('epoch')