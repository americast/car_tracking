import torch
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
import random
from skimage import io, transform
from skimage.transform import resize
import os
# import matplotlib.pyplot as plt
import pudb
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
from copy import copy

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])

# def get_data_unet(data):
#     count = 0
#     for each in data:
#         try:
#             img = io.imread(os.path.join('../VeRi/VeRi_with_plate/'+each[0].split("/")[-2], each[0].split("/")[-1]))
#             plt.imshow(img)
#             count += 1
#         except:
#             continue

#     print(count)


class data_unet(Dataset):
    def __init__(self, data):
        data_here = copy(data)
        data_here.sort(key = lambda x: x[0])
        self.data = []
        pos = 0
        while True:
            if pos >= len(data_here):
                break
            prefix = data_here[pos][0].split("/")[-1].split("_")[0]
            temp_pos = pos
            while True:
                temp_pos += 1
                if temp_pos >= len(data_here):
                    break
                # pu.db
                prefix_here = data_here[temp_pos][0].split("/")[-1].split("_")[0]
                if prefix_here != prefix:
                    break
            for i in range(pos, temp_pos):
                for j in range(i + 1, temp_pos):
                    self.data.append((data_here[i], data_here[j]))
            pos = temp_pos


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("idx:" +str(idx))
        imgs = copy(self.data[idx])
        img_1, img_2 = 0, 0
        try:
            img_1 = io.imread(os.path.join('../VeRi/VeRi_with_plate/'+imgs[0][0].split("/")[-2], imgs[0][0].split("/")[-1]))
            img_2 = io.imread(os.path.join('../VeRi/VeRi_with_plate/'+imgs[1][0].split("/")[-2], imgs[1][0].split("/")[-1]))
        except:
            img_1 = np.zeros((256, 256, 3))
            img_2 = np.zeros((256, 256, 3))
        img_1 = resize(img_1, (256, 256, 3)).transpose((2,0,1))
        img_2 = resize(img_2, (256, 256, 3)).transpose((2,0,1))

        # print("problem is: "+str(imgs[0][-1]))
        view_1 = int(imgs[0][-1])
        view_2 = int(imgs[1][-1])
        R = self.create_rot_matrix(view_1, view_2)
        # pu.db
        # imgs[0][0] = torch.from_numpy(np.array(img_1))
        # imgs[1][0] = torch.from_numpy(np.array(img_2))
        # print(imgs[0][-1])
        # print(imgs[1][-1])
        # print(img_1.shape)
        # print(R.shape)
        # print(img_2.shape)
        return [img_1, R, img_2]


    def abs_angle(self, pos):
        if (pos == 0):
            return 0.
        elif (pos == 1):
            return np.pi
        elif (pos == 2):
            return np.pi / 2
        elif (pos == 3):
            return np.pi / 4
        elif (pos == 4):
            return 3 * np.pi / 4
        elif (pos == 5):
            return 3 * np.pi / 2
        elif (pos == 6):
            return 7 * np.pi / 4
        elif (pos == 7):
            return 5 * np.pi / 4
        else:
            print("No match :(")
            sys.exit(0)


    def get_angle_diff(self, init_pos, final_pos):
        # pu.db
        ang = - (self.abs_angle(final_pos) - self.abs_angle(init_pos))
        # if (ang < 0):
        #     ang = 2 * np.pi - ang
        return ang

    def create_rot_matrix(self, init_pos, final_pos):
        ang = self.get_angle_diff(init_pos, final_pos)
        R = np.array([[np.cos(ang), -np.sin(ang), 0, 0],\
                      [np.sin(ang),  np.cos(ang), 0, 0],\
                      [0, 0, 1, 0],\
                      [0, 0, 0, 1]])

        return R





def get_data(image_filenames, type_dict, files, input_shape, batch_size, tv = 't'):
    batch_x = image_filenames
    j = 0
    pos = True
    X = []
    Y = []
    count = 0
    while j < len(batch_x):
        # print( str(j)+"/"+str(len(batch_x)))
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
            # print( str(j)+"/"+str(len(batch_x)))
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
            # print( str(j)+"/"+str(len(batch_x)))
            label = [1.0]
            # print(("In datagen"))
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
        # print( str(j)+"/"+str(len(batch_x)))
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
        # print( (str(j)+"/"+str(len(batch_x))))
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
            # print( str(j)+"/"+str(len(batch_x)))
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
            # print( str(j)+"/"+str(len(batch_x)))
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
            print( str(j)+"/"+str(len(files_pair)))
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
                print(("\n\nResume iter\n\n"))

def get_data_hot_wild_ratio_pred_noninf(image_filenames, type_dict, files, files_pair, input_shape, batch_size, ratio = 0.1, tv = 't'):
    j = 0
    old_j = 0
    X = []
    Y = []
    count = 0
    # while True:
    #     j = 0
    while j < len(files_pair):
        print( str(j)+"/"+str(len(files_pair)))
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

        if (old_j != j or count >= len(files_pair)):
            X_arr = np.array(X)
            count = 0
            yield ([X_arr[:,0], X_arr[:,1]], np.array(Y))
            print(("\n\nResume iter\n\n"))
            old_j = j


# def get_data_hot_wild_ratio_pred_noniter(image_filenames, type_dict, files, files_pair, input_shape, batch_size, ratio = 0.1, tv = 't'):
    # j = 0
    # X = []
    # Y = []
    # count = 0
    # while j < len(files_pair):
    #     print( str(j)+"/"+str(len(files_pair)))
    #     label = [0,1]
    #     # if (count == 0):
    #     # pu.db
    #     if random.random() < ratio:
            
    #         each = files_pair[j][0]
    #         img =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each+".jpg", target_size = (input_shape[0], input_shape[1])))
    #         each_2 = files_pair[j][1]
    #         img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))

    #         j += 1

    #     else:
    #         ind_1 = 0
    #         ind_2 = 0
    #         while True:
    #             ind_1 = random.randrange(0,len(files))
    #             ind_2 = random.randrange(0,len(files))

    #             if ind_2 not in range(*type_dict[files[ind_1].split("/")[0]]) : break

    #         each = files[ind_1]
    #         img =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each+".jpg", target_size = (input_shape[0], input_shape[1])))
    #         each_2 = files[ind_2]
    #         img_2 =  image.img_to_array(image.load_img("../VeRI_Wild/images/images/"+each_2+".jpg", target_size = (input_shape[0], input_shape[1])))

    #         label = [1,0]
    #         # if label == [1,0]: pu.db

    #     count +=1
    #     # pu.db

    #     X.append(np.concatenate((preprocess_input(seq(images = img.reshape(1, input_shape[0], input_shape[1], 3))), preprocess_input(seq(images = img_2.reshape(1, input_shape[0], input_shape[1], 3)))), axis = 0))
    #     Y.append(label)



    #     # batch_y = labels[idx * batch_size:(idx + 1) * batch_size]

    #     if (count == batch_size or count >= len(files_pair)):
    #         X_arr = np.array(X)
    #         count = 0
    #         yield ([X_arr[:,0], X_arr[:,1]], np.array(Y))
    #         print(("\n\nResume iter\n\n"))