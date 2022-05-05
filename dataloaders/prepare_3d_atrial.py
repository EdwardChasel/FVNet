import glob
import os
import random
import numpy as np
import nrrd
from functools import partial

file_path = 'D:/Datasets/medical/AtrailSeg_2018/AtrialSeg_2018'
# save_path = 'E:/dataset/medical_data/Atrial/AtrialSeg_3d_npy_256'
save_path = 'E:/dataset/medical_data/Atrial/AtrialSeg_3d_npy_margin'

def nrrd_to_npy(nrrd_path):
    data, info = nrrd.read(nrrd_path)
    # convert to numpy
    data = np.array(data, dtype=float)
    return data, info

def FixedCropHeart(image, label, heart_range, crop_size=[384, 384, 80]):
    crop_center = [heart_range[0] + np.floor((heart_range[1] - heart_range[0])/2), 
                    heart_range[2] + np.floor((heart_range[3] - heart_range[2])/2), 
                    heart_range[4] + np.floor((heart_range[5] - heart_range[4])/2)]
    crop_range_x = [int(crop_center[0] - np.floor(crop_size[0]/2)), int(crop_center[0] + np.floor(crop_size[0]/2))]
    crop_range_y = [int(crop_center[1] - np.floor(crop_size[1]/2)), int(crop_center[1] + np.floor(crop_size[1]/2))]
    crop_range_z = [int(crop_center[2] - np.floor(crop_size[2]/2)), int(crop_center[2] + np.floor(crop_size[2]/2))]
    # Padding
    pad_num_x = (int(max(-crop_range_x[0], 0)), int(max(crop_range_x[1]-image.shape[0], 0)))
    pad_num_y = (int(max(-crop_range_y[0], 0)), int(max(crop_range_y[1]-image.shape[1], 0)))
    pad_num_z = (int(max(-crop_range_z[0], 0)), int(max(crop_range_z[1]-image.shape[2], 0)))
    image = np.pad(image, [pad_num_x, pad_num_y, pad_num_z], mode='constant', constant_values=0)
    label = np.pad(label, [pad_num_x, pad_num_y, pad_num_z], mode='constant', constant_values=0)
    # Crop
    image = image[crop_range_x[0] + pad_num_x[0]:crop_range_x[1] + pad_num_x[0], 
                    crop_range_y[0] + pad_num_y[0]:crop_range_y[1] + pad_num_y[0], 
                    crop_range_z[0] + pad_num_z[0]:crop_range_z[1] + pad_num_z[0]]
    label = label[crop_range_x[0] + pad_num_x[0]:crop_range_x[1] + pad_num_x[0], 
                    crop_range_y[0] + pad_num_y[0]:crop_range_y[1] + pad_num_y[0], 
                    crop_range_z[0] + pad_num_z[0]:crop_range_z[1] + pad_num_z[0]]
    return image, label

def MarginCropHeart(image, label, heart_range, crop_size=[112, 112, 80]):
    w, h, d = label.shape
    px = max(crop_size[0] - (heart_range[1] - heart_range[0]), 0) // 2
    py = max(crop_size[1] - (heart_range[3] - heart_range[2]), 0) // 2
    pz = max(crop_size[2] - (heart_range[5] - heart_range[4]), 0) // 2
    minx = max(heart_range[0] - np.random.randint(10, 20) - px, 0)
    maxx = min(heart_range[1] + np.random.randint(10, 20) + px, w)
    miny = max(heart_range[2] - np.random.randint(10, 20) - py, 0)
    maxy = min(heart_range[3] + np.random.randint(10, 20) + py, h)
    minz = max(heart_range[4] - np.random.randint(5, 10) - pz, 0)
    maxz = min(heart_range[5] + np.random.randint(5, 10) + pz, d)

    image = (image - np.mean(image)) / np.std(image)
    image = image.astype(np.float32)
    image = image[minx:maxx, miny:maxy, minz:maxz]
    label = label[minx:maxx, miny:maxy, minz:maxz]
    return image, label

def processing_data(file_path, train_size=0.8, val_size=0.2, test_size=0, seed=2020, heart_crop='fixed'):
    print('Start processing!')
    filelist = glob.glob(os.path.join(file_path, '*', 'lgemri.nrrd'))

    r = random.random
    random.seed(seed)
    random.shuffle(filelist, random=r)

    if train_size + val_size + test_size != 1:
        raise ValueError('The sum of train_size, val_size and test_size must be 1 !!!')
    
    N = len(filelist)
    train_size = round(N * train_size)
    val_size = round(N * val_size)
    test_size = round(N * test_size)

    for i in range(N):
        image, _ = nrrd_to_npy(filelist[i])
        label, _ = nrrd_to_npy(filelist[i].replace('lgemri', 'laendo'))
        imagename = os.path.split(os.path.split(filelist[i])[0])[1] + '_lgemri'
        labelname = imagename.replace('lgemri', 'laendo')

        if i < train_size:
            current_path = os.path.join(save_path, 'train')
        elif train_size <= i < (train_size + val_size):
            current_path = os.path.join(save_path, 'val')
        else:
            current_path = os.path.join(save_path, 'test')

        if not os.path.exists(current_path):
            os.makedirs(current_path)

        # Location information of the heart
        heart_xx, heart_yy, heart_zz = np.where(label > 0)
        heart_range_x = [np.min(heart_xx), np.max(heart_xx)]
        heart_range_y = [np.min(heart_yy), np.max(heart_yy)]
        heart_range_z = [np.min(heart_zz), np.max(heart_zz)]
        heart_range = heart_range_x + heart_range_y + heart_range_z

        # Crop image and label across the heart
        if heart_crop == 'fixed':
            print('fixed')
            image, label = FixedCropHeart(image, label, heart_range, crop_size=[256, 256, 80])
        elif heart_crop == 'margin':
            print('margin')
            image, label = MarginCropHeart(image, label, heart_range, crop_size=[112, 112, 80])

        # Save image and label
        np.save(os.path.join(current_path, imagename + '.npy'), image.astype(np.float16))
        np.save(os.path.join(current_path, labelname + '.npy'), label.astype(np.float16))
        
        # Print the information
        heart_range_str = [str(i) for i in heart_range]
        heart_range_str = ' '.join(heart_range_str)
        current_info = imagename + ' heart_range ' + heart_range_str
        print(current_info)
        print('image.shape =', image.shape)
        print('label.shape =', label.shape)

        # Record the location information of the heart
        with open(os.path.join(current_path, 'data_info.txt'), 'a') as f:
            if os.path.getsize(os.path.join(current_path, 'data_info.txt')):
                f.write('\n')
            f.write(current_info)
            # e.g. 'volumn-0 heart_range 68 79 12 69 56 98'
    
    print('\nEnd of processing!')
    print('Total image:', N)

if __name__ == "__main__":
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    processing_data(file_path=file_path, train_size=0.8, val_size=0.2, test_size=0, seed=2020, heart_crop='margin')
