import keras
import tensorflow as tf
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, RMSprop
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import random
from copy import copy

class My_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        j = 0
        label = 1
        pos = True
        X = []
        Y = []
        while j < len(range(batch_x)):
            each = batch_x[j]
            img = resize(imread(each), (1, 200, 200))

            range_here = type_dict[img]
            if pos:
                while True:
                    ind = random.randrange(*range_here)
                    each_2 = files[ind]
                    if (each_2 != each): break
                
                img_2 = resize(imread(each_2), (1, 200, 200))
                pos = False
            else:
                
                while True:
                    ind = random.randrange(0,len(files))
                    if ind not in range(*range_here): break

                
                label = 0
                pos = True
                j+=1

            X.append(np.concatenate((img, img_2), axis = 0))
            Y.append(label)



        # batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(X), np.array(Y)


def create_base_network(in_dim):
    """ Base network to be shared (eq. to feature extraction).
    """
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    input_shape=(200, 200)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))

    return model


files = os.listdir("../VeRi/VeRi_with_plate/image_train")
files_perm = copy(files)
random.shuffle(files_perm)
files_first_name = [f.split("_")[0] for f in files]
type_dict = {}
i_start = 0
name_now = files_first_name[0]
for i in range(len(files_first_name)):
    if(files_first_name[i] != name_now):
        type_dict[name_now]=(i_start, i)
        name_now = files_first_name[i]
        i_start = i + 1

type_dict[files_first_name[-1]] = len(files_first_name)

training_filenames = []
GT_training = []
batch_size = 1

validation_filenames = []
GT_validation = []

my_training_batch_generator = My_Generator(files_perm, GT_training, batch_size)

input1 = Sequential()
input2 = Sequential()
input1.add(Layer(input_shape=(200, 200)))
input2.add(Layer(input_shape=(200, 200)))

base_network = create_base_network(in_dim)
add_shared_layer(base_network, [input1, input2])


# my_validation_batch_generator = My_Generator(validation_filenames, GT_validation, batch_size)

model.fit_generator(generator=my_training_batch_generator,
                                        steps_per_epoch=(num_training_samples // batch_size),
                                        epochs=num_epochs,
                                        verbose=1,
                                        validation_data=my_validation_batch_generator,
                                        validation_steps=(num_validation_samples // batch_size),
                                        use_multiprocessing=True,
                                        workers=16,
                                        max_queue_size=32)



