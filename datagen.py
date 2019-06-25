from skimage.io import imread
from skimage.transform import resize
import numpy as np
import random

def get_data(image_filenames, type_dict, files, input_shape, batch_size, tv = 't'):
    batch_x = image_filenames
    j = 0
    pos = True
    X = []
    Y = []
    count = 0
    while j < len(batch_x):
        # print str(j)+"/"+str(len(batch_x))
        label = 0
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
            
            label = 1
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

def get_data_hot(image_filenames, type_dict, files, input_shape, batch_size, tv = 't'):
    batch_x = image_filenames
    j = 0
    pos = True
    X = []
    Y = []
    count = 0
    while j < len(batch_x):
        # print str(j)+"/"+str(len(batch_x))
        label = [0,1]
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
            
            label = [1,0]
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

def get_data_ratio(image_filenames, type_dict, files, input_shape, batch_size, ratio = 0.1, tv = 't'):
    batch_x = image_filenames
    j = 0
    X = []
    Y = []
    count = 0
    while j < len(batch_x):
        # print str(j)+"/"+str(len(batch_x))
        label = 0
        if (count == 0):
            X = []
            Y = []
        each = batch_x[j]
        if tv == 't':
            img = resize(imread("../VeRi/VeRi_with_plate/image_train/"+each), (input_shape[0], input_shape[1]))
        else:
            img = resize(imread("../VeRi/VeRi_with_plate/image_test/"+each), (input_shape[0], input_shape[1]))
        # pu.db

        range_here = type_dict[each.split("_")[0]]
        count +=1
        if random.random() <= ratio:
            while True:
                ind = random.randrange(*range_here)
                each_2 = files[ind]
                if (each_2 != each): break
            
            if tv == 't':
                img_2 = resize(imread("../VeRi/VeRi_with_plate/image_train/"+each_2), (input_shape[0], input_shape[1]))
            else:
                img_2 = resize(imread("../VeRi/VeRi_with_plate/image_test/"+each_2), (input_shape[0], input_shape[1]))

        else:
            
            while True:
                ind = random.randrange(0,len(files))
                if ind not in range(*range_here): break

            each_2 = files[ind]
            if tv == 't':
                img_2 = resize(imread("../VeRi/VeRi_with_plate/image_train/"+each_2), (input_shape[0], input_shape[1]))
            else:
                img_2 = resize(imread("../VeRi/VeRi_with_plate/image_test/"+each_2), (input_shape[0], input_shape[1]))
            
            label = 1
        j+=1

        # pu.db

        X.append(np.concatenate((img.reshape(1, input_shape[0], input_shape[1], 3), img_2.reshape(1, input_shape[0], input_shape[1], 3)), axis = 0))
        Y.append(label)



        # batch_y = labels[idx * batch_size:(idx + 1) * batch_size]

        if (count == batch_size):
            X_arr = np.array(X)
            count = 0
            yield ([X_arr[:,0], X_arr[:,1]], np.array(Y))
