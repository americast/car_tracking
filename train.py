choice = raw_input("Would you like to load an existing model? (<model_file_name>/n): ")
tv = raw_input("Train or validate? (t/v): ")
from keras.models import Sequential
from keras.layers import *
from keras.layers.core import *
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.models import *

from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import *
from keras.layers.core import *
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD, RMSprop
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os
import random
from copy import copy
import pudb

EPOCHS = 10000
batch_size = 56
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

input_shape = (224, 224, 3)

def get_data(image_filenames, type_dict, files, tv = 't'):
    batch_x = image_filenames
    j = 0
    pos = True
    X = []
    Y = []
    count = 0
    while j < len(batch_x):
        # print str(j)+"/"+str(len(batch_x))
        label = 1
        if (count == 0 and pos):
            X = []
            Y = []
        each = batch_x[j]
        if tv == 't':
            img = resize(imread("../VeRi/VeRi_with_plate/image_train/"+each), (input_shape[0], input_shape[1]))
        else:
            img = resize(imread("../VeRi/VeRi_with_plate/image_test/"+each), (input_shape[0], input_shape[1]))
        # pu.db

        range_here = type_dict[each.split("_")[0]]
        if pos:
            while True:
                ind = random.randrange(*range_here)
                each_2 = files[ind]
                if (each_2 != each): break
            
            if tv == 't':
                img_2 = resize(imread("../VeRi/VeRi_with_plate/image_train/"+each_2), (input_shape[0], input_shape[1]))
            else:
                img_2 = resize(imread("../VeRi/VeRi_with_plate/image_test/"+each_2), (input_shape[0], input_shape[1]))
            pos = False
        else:
            count +=1
            
            while True:
                ind = random.randrange(0,len(files))
                if ind not in range(*range_here): break

            each_2 = files[ind]
            if tv == 't':
                img_2 = resize(imread("../VeRi/VeRi_with_plate/image_train/"+each_2), (input_shape[0], input_shape[1]))
            else:
                img_2 = resize(imread("../VeRi/VeRi_with_plate/image_test/"+each_2), (input_shape[0], input_shape[1]))
            
            label = 0
            pos = True
            j+=1

        # pu.db

        X.append(np.concatenate((img.reshape(1, input_shape[0], input_shape[1], 3), img_2.reshape(1, input_shape[0], input_shape[1], 3)), axis = 0))
        Y.append(label)



        # batch_y = labels[idx * batch_size:(idx + 1) * batch_size]

        if (count == batch_size):
            X_arr = np.array(X)
            count = 0
            yield ([X_arr[:,0], X_arr[:,1]], np.array(Y))

            

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_base_network(in_dim):
    """ Base network to be shared (eq. to feature extraction).
    """
    # input = Input(shape=input_shape)

    # x = Conv2D(8, kernel_size=(5, 5), strides=(1, 1),
    #                 activation='relu',
    #                 input_shape=(input_shape[0], input_shape[1]))(input)

    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    # x = Conv2D(16, (5, 5), activation='relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Flatten()(x)
    # x = Dense(1000, activation='relu')(x)
    # return Model(input, x)
    model = ResNet50(weights="imagenet")
    print(model.summary())
    return Model(inputs=model.input, outputs=model.get_layer('fc1000').output)


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

#### Training images
files = os.listdir("../VeRi/VeRi_with_plate/image_train")
files.sort()
# files = files[:160]
files_perm = copy(files)
random.shuffle(files_perm)
files_first_name = [f.split("_")[0] for f in files]
type_dict = {}
i_start = 0
name_now = files_first_name[0]
for i in range(1, len(files_first_name)):
    if(files_first_name[i] != name_now):
        print(name_now, i_start, i)
        type_dict[name_now]=(i_start, i)
        name_now = files_first_name[i]
        i_start = i

type_dict[files_first_name[-1]] = (i_start, len(files_first_name))

#### Validation images
files_val = os.listdir("../VeRi/VeRi_with_plate/image_test")
files_val.sort()
# pu.db
files_val = files_val
files_perm_val = copy(files_val)
random.shuffle(files_perm_val)
files_first_name_val = [f_val.split("_")[0] for f_val in files_val]
type_dict_val = {}
i_start_val = 0
name_now_val = files_first_name_val[0]
for i_val in range(1, len(files_first_name_val)):
    if(files_first_name_val[i_val] != name_now_val):
        print(name_now_val, i_start_val, i_val)
        type_dict_val[name_now_val]=(i_start_val, i_val)
        name_now_val = files_first_name_val[i_val]
        i_start_val = i_val

type_dict_val[files_first_name_val[-1]] = (i_start_val, len(files_first_name_val))


# pu.db
training_filenames = []
GT_training = []

validation_filenames = []
GT_validation = []

# my_training_batch_generator = My_Generator(files_perm, GT_training, batch_size)
# print("Would you like to load an existing model? (y/n)")
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)
if choice != 'n' and choice != 'N':
    model.load_weights(choice)
model_gpu = multi_gpu_model(model, gpus=4)
#model_gpu = model

# train
rms = RMSprop()
model_gpu.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])


# my_validation_batch_generator = My_Generator(validation_filenames, GT_validation, batch_size)
num_training_samples = len(files)
num_validation_samples = len(files_val)
# for x,y in get_data(files_perm):
#     pu.db

#     break
model_gpu.summary()


checkpointer = ModelCheckpoint(monitor='acc', filepath="check.h5", verbose=True,
                                   save_best_only = True)
print("Here")
# pu.db
if tv == 'v' or tv == 'V':
    print(model_gpu.evaluate_generator(get_data(files_perm_val, type_dict_val, files_val, 'v'), steps=(num_validation_samples * 2 // (batch_size * 4)), use_multiprocessing=True, workers=16, max_queue_size=32))
else:
    for _ in xrange(EPOCHS):
        print _
        # if _ % 50 == 0:
        #     for x in get_data(files_perm):
        #       print "Prediction: "
        #       out = model.predict(x[0])
        #       print("out: "+str(out))
        #       print("orig: "+str(x[1]))

            
        model_gpu.fit_generator(generator=get_data(files_perm, type_dict, files),
                                            steps_per_epoch=(num_training_samples * 2 // (batch_size * 4)),
                                            epochs=1,
                                            verbose=1,
                                            validation_data=get_data(files_perm_val, type_dict_val, files_val, 'v'),
                                            validation_steps=(num_validation_samples // batch_size * 4),
                                            use_multiprocessing=True,
                                            workers=16,
                                            max_queue_size=32)

        model.save_weights("check_weights.h5")
        #model_gpu.save("check.h5")



