from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
import random

from imgaug import augmenters as iaa
seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])


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
            img =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_train/"+each, target_size = (input_shape[0], input_shape[1])))
        else:
            img =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_test/"+each, target_size = (input_shape[0], input_shape[1])))
        # pu.db

        range_here = type_dict[each.split("_")[0]]
        if pos:
            while True:
                ind = random.randrange(*range_here)
                each_2 = files[ind]
                if (each_2 != each): break
            
            if tv == 't':
                img_2 =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_train/"+each_2, target_size = (input_shape[0], input_shape[1])))
            else:
                img_2 =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_test/"+each_2, target_size = (input_shape[0], input_shape[1])))
            pos = False
        else:
            count +=1
            
            while True:
                ind = random.randrange(0,len(files))
                if ind not in range(*range_here): break

            each_2 = files[ind]
            if tv == 't':
                img_2 =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_train/"+each_2, target_size = (input_shape[0], input_shape[1])))
            else:
                img_2 =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_test/"+each_2, target_size = (input_shape[0], input_shape[1])))
            
            label = 1
            pos = True
            j+=1

        # pu.db

        X.append(np.concatenate((preprocess_input(img.reshape(1, input_shape[0], input_shape[1], 3)), preprocess_input(img_2.reshape(1, input_shape[0], input_shape[1], 3))), axis = 0))
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
    while True:
        j = 0
        while j < len(batch_x):
            # print str(j)+"/"+str(len(batch_x))
            label = [0,1]
            if (count == 0 and pos):
                X = []
                Y = []
            each = batch_x[j]
            if tv == 't':
                img =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_train/"+each, target_size = (input_shape[0], input_shape[1])))
            else:
                img =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_test/"+each, target_size = (input_shape[0], input_shape[1])))
            # pu.db

            range_here = type_dict[each.split("_")[0]]
            if pos:
                while True:
                    ind = random.randrange(*range_here)
                    each_2 = files[ind]
                    if (each_2 != each): break
                
                if tv == 't':
                    img_2 =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_train/"+each_2, target_size = (input_shape[0], input_shape[1])))
                else:
                    img_2 =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_test/"+each_2, target_size = (input_shape[0], input_shape[1])))
                pos = False
            else:
                count +=1
                
                while True:
                    ind = random.randrange(0,len(files))
                    if ind not in range(*range_here): break

                each_2 = files[ind]
                if tv == 't':
                    img_2 =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_train/"+each_2, target_size = (input_shape[0], input_shape[1])))
                else:
                    img_2 =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_test/"+each_2, target_size = (input_shape[0], input_shape[1])))
                
                label = [1,0]
                pos = True
                j+=1

            # pu.db

            X.append(np.concatenate((preprocess_input(img.reshape(1, input_shape[0], input_shape[1], 3)), preprocess_input(img_2.reshape(1, input_shape[0], input_shape[1], 3))), axis = 0))
            Y.append(label)



            # batch_y = labels[idx * batch_size:(idx + 1) * batch_size]

            if (count == batch_size):
                X_arr = np.array(X)
                count = 0
                yield ([X_arr[:,0], X_arr[:,1]], np.array(Y))

def get_data_hot_unit(image_filenames, type_dict, files, input_shape, batch_size, tv = 't'):
    batch_x = image_filenames
    j = 0
    X = []
    Y = []
    count = 0
    while True:
        j = 0
        while j < len(batch_x):
            # print str(j)+"/"+str(len(batch_x))
            label = [1.0]
            # print("In datagen")
            if (count == 0):
                X = []
                Y = []
            each = batch_x[j]
            if tv == 't':
                img =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/unit/"+each, target_size = (input_shape[0], input_shape[1])))
            else:
                img =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/unit/"+each, target_size = (input_shape[0], input_shape[1])))
            # pu.db
            
            # pu.db
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = np.concatenate((img, img, img), axis = -1)

            X.append(img.reshape(input_shape[0], input_shape[1], 3))
            num = each.split("_")[0]
            if num == "0":
                label = [1, 0, 0]
            elif num == "1":     
                label = [0, 1, 0]
            else:
                label = [0, 0, 1]
            Y.append(label)

            count+=1
            j+=1

            # batch_y = labels[idx * batch_size:(idx + 1) * batch_size]

            if (count == batch_size):
                X_arr = np.array(X)
                count = 0
                yield X_arr.astype("float32"), np.array(Y).astype("float32")



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
            img =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_train/"+each, target_size = (input_shape[0], input_shape[1])))
        else:
            img =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_test/"+each, target_size = (input_shape[0], input_shape[1])))
        # pu.db

        range_here = type_dict[each.split("_")[0]]
        count +=1
        if random.random() <= ratio:
            while True:
                ind = random.randrange(*range_here)
                each_2 = files[ind]
                if (each_2 != each): break
            
            if tv == 't':
                img_2 =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_train/"+each_2, target_size = (input_shape[0], input_shape[1])))
            else:
                img_2 =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_test/"+each_2, target_size = (input_shape[0], input_shape[1])))

        else:
            
            while True:
                ind = random.randrange(0,len(files))
                if ind not in range(*range_here): break

            each_2 = files[ind]
            if tv == 't':
                img_2 =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_train/"+each_2, target_size = (input_shape[0], input_shape[1])))
            else:
                img_2 =  image.img_to_array(image.load_img("../VeRi/VeRi_with_plate/image_test/"+each_2, target_size = (input_shape[0], input_shape[1])))
            
            label = 1
        j+=1

        # pu.db

        X.append(np.concatenate((preprocess_input(img.reshape(1, input_shape[0], input_shape[1], 3)), preprocess_input(img_2.reshape(1, input_shape[0], input_shape[1], 3))), axis = 0))
        Y.append(label)



        # batch_y = labels[idx * batch_size:(idx + 1) * batch_size]

        if (count == batch_size):
            X_arr = np.array(X)
            count = 0
            yield ([X_arr[:,0], X_arr[:,1]], np.array(Y))

def get_data_ratio_wild(image_filenames, type_dict, files, input_shape, batch_size, ratio = 0.1, tv = 't'):
    batch_x = image_filenames
    j = 0
    X = []
    Y = []
    count = 0
    while j < len(batch_x):
        # print (str(j)+"/"+str(len(batch_x)))
        label = 0
        if (count == 0):
            X = []
            Y = []
        each = batch_x[j]
        if tv == 't':
            img =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each+".jpg", target_size = (input_shape[0], input_shape[1])))
        else:
            img =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each+".jpg", target_size = (input_shape[0], input_shape[1])))
        # pu.db

        range_here = type_dict[each.split("/")[0]]
        count +=1
        if random.random() <= ratio:
            while True:
                ind = random.randrange(*range_here)
                each_2 = files[ind]
                if (each_2 != each): break
            
            if tv == 't':
                img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))
            else:
                img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))

        else:
            
            while True:
                ind = random.randrange(0,len(files))
                if ind not in range(*range_here): break

            each_2 = files[ind]
            if tv == 't':
                img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))
            else:
                img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))
            
            label = 1
        j+=1

        # pu.db

        X.append(np.concatenate((preprocess_input(img.reshape(1, input_shape[0], input_shape[1], 3)), preprocess_input(img_2.reshape(1, input_shape[0], input_shape[1], 3))), axis = 0))
        Y.append(label)



        # batch_y = labels[idx * batch_size:(idx + 1) * batch_size]

        if (count == batch_size):
            X_arr = np.array(X)
            count = 0
            yield ([X_arr[:,0], X_arr[:,1]], np.array(Y))


def get_data_hot_wild(image_filenames, type_dict, files, input_shape, batch_size, tv = 't'):
    batch_x = image_filenames
    j = 0
    pos = True
    X = []
    Y = []
    count = 0
    while True:
        j = 0
        while j < len(batch_x):
            # print str(j)+"/"+str(len(batch_x))
            label = [0,1]
            if (count == 0 and pos):
                X = []
                Y = []
            each = batch_x[j]
            if tv == 't':
                img =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each+".jpg", target_size = (input_shape[0], input_shape[1])))
            else:
                img =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each+".jpg", target_size = (input_shape[0], input_shape[1])))
            # pu.db

            range_here = type_dict[each.split("/")[0]]
            if pos:
                while True:
                    ind = random.randrange(*range_here)
                    each_2 = files[ind]
                    if (each_2 != each): break
                
                if tv == 't':
                    img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))
                else:
                    img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))
                pos = False
            else:
                count +=1
                
                while True:
                    ind = random.randrange(0,len(files))
                    if ind not in range(*range_here): break

                each_2 = files[ind]
                if tv == 't':
                    img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))
                else:
                    img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))
                
                label = [1,0]
                pos = True
                j+=1

            # pu.db

            X.append(np.concatenate((preprocess_input(img.reshape(1, input_shape[0], input_shape[1], 3)), preprocess_input(img_2.reshape(1, input_shape[0], input_shape[1], 3))), axis = 0))
            Y.append(label)



            # batch_y = labels[idx * batch_size:(idx + 1) * batch_size]

            if (count == batch_size):
                X_arr = np.array(X)
                count = 0
                yield ([X_arr[:,0], X_arr[:,1]], np.array(Y))


def get_data_hot_wild_ratio(image_filenames, type_dict, files, files_pair, input_shape, batch_size, ratio = 0.1, tv = 't'):
    j = 0
    X = []
    Y = []
    count = 0
    while True:
        j = 0
        while j < len(files_pair):
            # print str(j)+"/"+str(len(batch_x))
            label = [0,1]
            if (count == 0):
                X = []
                Y = []
            if random.random() < ratio:
                
                each = files_pair[j][0]
                img =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each+".jpg", target_size = (input_shape[0], input_shape[1])))
                each_2 = files_pair[j][1]
                img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))

                j += 1

            else:
                ind_1 = 0
                ind_2 = 0
                while True:
                    ind_1 = random.randrange(0,len(files))
                    ind_2 = random.randrange(0,len(files))

                    if ind_2 not in range(*type_dict[files[ind_1].split("/")[0]]) : break

                each = files[ind_1]
                img =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each+".jpg", target_size = (input_shape[0], input_shape[1])))
                each_2 = files[ind_2]
                img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))

                label = [1,0]

            count +=1
            # pu.db

            X.append(np.concatenate((preprocess_input(seq(images = img.reshape(1, input_shape[0], input_shape[1], 3))), preprocess_input(seq(images = img_2.reshape(1, input_shape[0], input_shape[1], 3)))), axis = 0))
            Y.append(label)



            # batch_y = labels[idx * batch_size:(idx + 1) * batch_size]

            if (count == batch_size or count >= len(files_pair)):
                X_arr = np.array(X)
                count = 0
                yield ([X_arr[:,0], X_arr[:,1]], np.array(Y))


def get_data_hot_wild_ratio_pred(image_filenames, type_dict, files, files_pair, input_shape, batch_size, ratio = 0.1, tv = 't'):
    j = 0
    X = []
    Y = []
    count = 0
    while True:
        j = 0
        while j < len(files_pair):
            print str(j)+"/"+str(len(files_pair))
            label = [0,1]
            if (count == 0):
                X = []
                Y = []
            if random.random() < ratio:
                
                each = files_pair[j][0]
                img =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each+".jpg", target_size = (input_shape[0], input_shape[1])))
                each_2 = files_pair[j][1]
                img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))

                j += 1

            else:
                ind_1 = 0
                ind_2 = 0
                while True:
                    ind_1 = random.randrange(0,len(files))
                    ind_2 = random.randrange(0,len(files))

                    if ind_2 not in range(*type_dict[files[ind_1].split("/")[0]]) : break

                each = files[ind_1]
                img =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each+".jpg", target_size = (input_shape[0], input_shape[1])))
                each_2 = files[ind_2]
                img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))

                label = [1,0]

            count +=1
            # pu.db

            X.append(np.concatenate((preprocess_input(seq(images = img.reshape(1, input_shape[0], input_shape[1], 3))), preprocess_input(seq(images = img_2.reshape(1, input_shape[0], input_shape[1], 3)))), axis = 0))
            Y.append(label)



            # batch_y = labels[idx * batch_size:(idx + 1) * batch_size]

            if (count == batch_size or count >= len(files_pair)):
                X_arr = np.array(X)
                count = 0
                yield [X_arr[:,0], X_arr[:,1]]                