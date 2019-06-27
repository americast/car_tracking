choice = raw_input("Would you like to load an existing model? (<model_file_name>/n): ")
tv = raw_input("Train or validate? (t/v): ")
from keras.models import Sequential
from keras.layers import *
from keras.layers.core import *
from keras.optimizers import SGD, RMSprop
from keras import backend as K
from keras.models import *
from keras import losses

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
import sys
import random
from copy import copy
import pudb
from datagen import *
from utils import *

EPOCHS = 10000
batch_size = 56
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

input_shape = (224, 224, 3)

from keras.legacy import interfaces
import keras.backend as K
from keras.optimizers import Optimizer

class Adam_lr_mult(Optimizer):
    """Adam optimizer.
    Adam optimizer, with learning rate multipliers built on Keras implementation
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        
    AUTHOR: Erik Brorson (https://erikbrorson.github.io/2018/04/30/Adam-with-learning-rate-multipliers/)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False,
                 multipliers=None, debug_verbose=False,**kwargs):
        super(Adam_lr_mult, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.multipliers = multipliers
        self.debug_verbose = debug_verbose

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):

            # Learning rate multipliers
            if self.multipliers:
                multiplier = [mult for mult in self.multipliers if mult in p.name]
            else:
                multiplier = None
            if multiplier:
                new_lr_t = lr_t * self.multipliers[multiplier[0]]
                if self.debug_verbose:
                    print('Setting {} to learning rate {}'.format(multiplier[0], new_lr_t))
                    print(K.get_value(new_lr_t))
            else:
                new_lr_t = lr_t
                if self.debug_verbose:
                    print('No change in learning rate {}'.format(p.name))
                    print(K.get_value(new_lr_t))
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - new_lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - new_lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'multipliers':self.multipliers}
        base_config = super(Adam_lr_mult, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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
    # print(model.summary())
    return Model(inputs=model.input, outputs=model.get_layer('fc1000').output)



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
# files_val = files_val[:160]
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

vect_cat = Concatenate(axis = -1, name = "cat_layer")([processed_a, processed_b])
hid_cat_2 = Dense(512, activation='relu', name="cat_dense_1")(vect_cat)
hid_cat_1 = Dense(256, activation='relu', name="cat_dense_2")(hid_cat_2)
hid_cat = Dense(64, activation='relu', name="cat_dense_3")(hid_cat_1)
out = Dense(2, activation='softmax', name="cat_dense_out")(hid_cat)


# distance = Lambda(euclidean_distance,
#                 output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], out)
if choice != 'n' and choice != 'N':
    model.load_weights(choice)
# model_gpu = multi_gpu_model(model, gpus=4)
model_gpu = model

# train
learning_rate_multipliers = {"cat_dense_1": 10, "cat_dense_2": 10, "cat_dense_3": 10, "cat_dense_out": 10}

adam_with_lr_multipliers = Adam_lr_mult(lr = 0.0001, multipliers=learning_rate_multipliers)
model_gpu.compile(loss=losses.categorical_crossentropy, optimizer=adam_with_lr_multipliers, metrics=["accuracy"])


# my_validation_batch_generator = My_Generator(validation_filenames, GT_validation, batch_size)
num_training_samples = len(files)
num_validation_samples = len(files_val)
# for x,y in get_data(files_perm):
#     pu.db

#     break
print(model_gpu.summary())
# sys.exit(0)



checkpointer = ModelCheckpoint(monitor='acc', filepath="check_weights.h5", verbose=True,
                                   save_best_only = True)
print("Here")
# pu.db
if tv == 'v' or tv == 'V':
    print(model_gpu.evaluate_generator(get_data_hot(files_perm_val, type_dict_val, files_val, input_shape, batch_size, 'v'), steps=(num_validation_samples * 2 // (batch_size * 4)), use_multiprocessing=True, workers=16, max_queue_size=32))
else:
    for _ in xrange(EPOCHS):
        print _
        # if _ >= 0:
        #     for x in get_data_hot(files_perm_val, type_dict_val, files_val, input_shape, batch_size, 'v'):
        #       print "Prediction: "
        #       out = model.predict(x[0])
        #       print("out: "+str(out))
        #       print("orig: "+str(x[1]))
        #       break

            
        model_gpu.fit_generator(generator=get_data_hot(files_perm, type_dict, files, input_shape, batch_size),
                                            steps_per_epoch=(num_training_samples * 2 // (batch_size * 4)),
                                            epochs=1,
                                            verbose=1,
                                            # validation_data=get_data_hot(files_perm_val, type_dict_val, files_val, input_shape, batch_size,'v'),
                                            # validation_steps=(num_validation_samples // batch_size * 4),
                                            use_multiprocessing=True,
                                            workers=16,
                                            max_queue_size=32)

        # model.save_weights("check_weights_class.h5")
        pu.db
        model_gpu.save("check_weights_class.h5")