# choice = raw_input("Would you like to load an existing model? (<model_file_name>/n): ")
choice = "check_weights_class_wild_ratio.h5"
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
from copy import *
import pudb
from datagen import *
from utils import *
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import auc

EPOCHS = 10000
batch_size = 200
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
f = open("../VeRI_Wild/train_test_split/train_list.txt", "r")
files = []
while True:
    l = f.readline()
    if not l:
        f.close()
        break
    files.append(l.strip())
files.sort()
files = files[:100]
files_perm = copy(files)
random.shuffle(files_perm)
files_first_name = [f.split("/")[0] for f in files]
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

files_all_pairs = []
for i in range(len(files)):
    range_here = type_dict[files[i].split("/")[0]]
    for j in range(range_here[0], range_here[1]):
        if i == j: continue
        files_all_pairs.append(tuple(sorted([files[i], files[j]])))
# pu.db
files_all_pairs = list(set(files_all_pairs))
random.shuffle(files_all_pairs)

#### Validation images
f_val = open("../VeRI_Wild/train_test_split/test_3000.txt", "r")
files_val = []
while True:
    l_val = f_val.readline()
    if not l_val:
        f_val.close()
        break
    files_val.append(l_val.strip())
files_val = files_val[:100]
files_val.sort()
# pu.db
files_perm_val = copy(files_val)
random.shuffle(files_perm_val)
files_first_name_val = [f_val.split("/")[0] for f_val in files_val]
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

files_all_pairs_val = []
for i in range(len(files_val)):
    range_here = type_dict_val[files_val[i].split("/")[0]]
    for j in range(range_here[0], range_here[1]):
        if i == j: continue
        files_all_pairs_val.append(tuple(sorted([files_val[i], files_val[j]])))

files_all_pairs_val = list(set(files_all_pairs_val))
random.shuffle(files_all_pairs_val)


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
model_gpu = multi_gpu_model(model, gpus=4)
# model_gpu = model

# train
learning_rate_multipliers = {"cat_dense_1": 10, "cat_dense_2": 10, "cat_dense_3": 10, "cat_dense_out": 10}

adam_with_lr_multipliers = Adam_lr_mult(lr = 0.0001, multipliers=learning_rate_multipliers)
model_gpu.compile(loss=losses.categorical_crossentropy, optimizer=adam_with_lr_multipliers, metrics=["accuracy"])


# my_validation_batch_generator = My_Generator(validation_filenames, GT_validation, batch_size)
num_training_samples = len(files)
num_validation_samples = len(files_val)
# for x,y in get_data_hot_wild_ratio(files_perm, type_dict, files, files_all_pairs, input_shape, batch_size):
#     pu.db

#     break
print(model_gpu.summary())
# sys.exit(0)



# checkpointer = ModelCheckpoint(monitor='acc', filepath="check_weights.h5", verbose=True,
#                                    save_best_only = True)
print("Here")
acc_hist = 0.0
y_pred_net = []
y_org_net = []
# pu.db
if tv == 'v' or tv == 'V':
    print("Pred start")
    # avg_precisions = []
    precisions = []
    recalls = []
    k = []
    res_dict = {}
    # pu.db
    for data in get_data_hot_wild_ratio_pred_noninf(files_perm_val, type_dict_val, files_val, files_all_pairs_val, input_shape, batch_size, tv = 'v'):
        Y_pred = model_gpu.predict(data[0])
        y_pred = []
        for each in Y_pred:
            if each[1] > 0.3:
                y_pred.append(each[1])
            else:
                y_pred.append(0)
        # y_pred = np.argmax(Y_pred, axis=1)
        y_org = np.argmax(data[1], axis=1)
        y_pred_net.extend(y_pred)
        y_org_net.extend(y_org)
        for i in range(len(y_pred)):
            sum_here = sum(sum(sum(data[0][0][i])))
            if sum_here not in res_dict:
                res_dict[sum_here] = [[],[]]
            res_dict[sum_here][0].extend(y_pred)
            res_dict[sum_here][1].extend(y_org)
        # pu.db
        # ind_ones_pred = np.where(y_pred == 1)
        # ind_ones_pred_false = np.where(y_pred == 0)
        # ind_ones_org = np.where(y_org == 1)
        # ind_ones_org_false = np.where(y_org == 0)

        # TP = len(np.intersect1d(ind_ones_org, ind_ones_pred))
        # FP = len(np.intersect1d(ind_ones_org_false, ind_ones_pred))
        # TN = len(np.intersect1d(ind_ones_org_false, ind_ones_pred_false))
        # FN = len(np.intersect1d(ind_ones_org, ind_ones_pred_false))

        # try:
        #     precision = float(TP) / (TP + FP + 1e-06)
        # except:
        #     precision = 1.0
        # try:
        #     recall = float(TP) / (TP + FN + 1e-06)
        # except:
        #     recall = 0.0

        # Y_pred_one_only = Y_pred[y_pred == 1]
        # y_org_sorted = y_org[y_pred == 1]

        # # pu.db
        # try:
        #     Y_pred_one_only, y_org_sorted = zip(*sorted(zip(Y_pred_one_only, y_org_sorted), reverse = True))
        # except: pass

        # precisions_here = []
        # pred_good = 0

        # for i in range(len(Y_pred_one_only)):
        #     if (y_org_sorted[i] == 1):
        #         pred_good += 1
        #     precisions_here.append(float(pred_good)/(i+1))

        # try:
        #     avg_precisions.append(sum(precisions_here)/len(precisions_here))
        #     # if sum(precisions_here)/len(precisions_here) == 0:
        #     #     pu.db
        # except:
        #     avg_precisions.append(0.0)

        # precisions.append(precision)
        # recalls.append(recall)

    # pu.db
    aps = []
    for each in res_dict:
        y_pred = res_dict[each][0]
        y_org = res_dict[each][1]
        tot = [(y_pred[i], y_org[i]) for i in range(len(y_pred))]
        tot.sort(key = lambda x:x[0])
        y_pred = [x[0] for x in tot]
        y_org = [x[1] for x in tot]
        y_org_net_here = []
        y_pred_net_here = []
        tot_p = []
        tot_r = []
        for i in range(len(y_pred)):
            if y_pred > 0.3:
                y_pred_net_here.append(1)
            else:
                y_pred_net_here.append(0)
            # for j, _ in enumerate(y_pred_net_here):
            #     if _ > 0.3:
            #         y_pred_net_here[j] = 1
            y_org_net_here.append(y_org[i])
            scores = precision_recall_fscore_support(y_org_net_here, y_pred_net_here, average=None, labels=[1])
            tot_p.append(scores[0])
            tot_r.append(scores[1])
        tog = [(tot_p[i], tot_r[i]) for i in range(len(tot_p))]
        tog.sort(key = lambda x:x[1])
        tot_p = [x[0] for x in tog]
        tot_r = [x[1] for x in tog]
        try:
            k = auc(tot_r, tot_p)
            aps.append(k)
        except:
            pass




    # scores = precision_recall_fscore_support(y_org_net, y_pred_net, average=None, labels=[1])
    # precisions.append(scores[0])
    # recalls.append(scores[1])
    # tog = [(precisions[i], recalls[i]) for i in range(len(precisions))]
    # tog.sort(key=lambda x:x[1])
    # precisions = [x[0] for x in tog]
    # recalls = [x[1] for x in tog]
    pu.db
    # k = auc(recalls, precisions)
        # pu.db

else:
    for _ in range(EPOCHS):
        print(_)
        # if _ >= 0:
        #     for x in get_data_hot_wild_ratio(files_perm_val, type_dict_val, files_val, input_shape, batch_size, 'v'):
        #       print "Prediction: "
        #       out = model.predict(x[0])
        #       print("out: "+str(out))
        #       print("orig: "+str(x[1]))
        #       break

        for data in get_data_hot_wild_ratio_pred_noninf(files_perm_val, type_dict_val, files_val, files_all_pairs_val, input_shape, batch_size, tv = 't'):
            y_org = np.argmax(data[1], axis=1)
            zero_arr = np.zeros((10,10,3))
            x_starts = list(np.zeros((len(y_org))))
            y_starts = list(np.zeros((len(y_org))))
            y_pred_list = list(np.zeros((len(y_org))))
            for x_start in range(0, 224 - 10, 10):
                for y_start in range(0, 224 - 10, 10):
                    data_in = deepcopy(data[0])
                    for i, __ in enumerate(y_org):
                        if __ == 1:
                            img = data_in[0][i]
                            # pu.db
                            try:
                                img[0+x_start:10+x_start, 0+y_start:10+y_start, :] = zero_arr
                            except: pu.db
                    Y_pred = model_gpu.predict(data_in)
                    for i, __ in enumerate(y_org):
                        if __ == 1:
                            if Y_pred[i][1] > y_pred_list[i]:
                                x_starts[i] = x_start
                                y_starts[i] = y_start
                                y_pred_list[i] = Y_pred[i][1]

            for i, __ in enumerate(y_org):
                if __ == 1:
                    img = data[0][0][i]
                    img[0+x_starts[i]:10+x_starts[i], 0+y_starts[i]:10+y_starts[i], :] = zero_arr
            
            acc = model_gpu.fit(data[0], data[1])

        print()
        model.save_weights("check_weights_class_wild_ratio_occ.h5")
        print("Saving model")







        #     pu.db
        

            
        # acc = model_gpu.fit_generator(generator=get_data_hot_wild_ratio(files_perm, type_dict, files, files_all_pairs, input_shape, batch_size),
        #                                     steps_per_epoch=(num_training_samples * 2 // (batch_size * 4)),
        #                                     epochs=1,
        #                                     verbose=1,
        #                                     # validation_data=get_data_hot_wild_ratio(files_perm_val, type_dict_val, files_val, input_shape, batch_size,'v'),
        #                                     # validation_steps=(num_validation_samples // batch_size * 4),
        #                                     use_multiprocessing=True,
        #                                     workers=16,
        #                                     max_queue_size=32)
        # # pu.db
        # if acc.history['acc'][0] > acc_hist:
        #     print("Saving model")
        #     if choice.lower() == "n":
        #     else:
        #         model.save_weights(choice)

        #     acc_hist = acc.history['acc'][0]
        # # model.save_weights("check_weights_class.h5")
        # pu.db
        # model_gpu.save("check_weights_class.h5")