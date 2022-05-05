import glob
import os
import random
import numpy as np
import nibabel as nib
from functools import partial

file_path = 'D:/Datasets/medical/LITS_Data/lits'
save_path = 'E:/dataset/medical_data/LITS_Data/lits_npy'

def nii_to_npy(nii_path):
    data = nib.load(nii_path) # load nii image
    data = data.get_data() # convert nii to numpy
    return data

def processing_data(file_path, train_size=0.8, val_size=0.2, test_size=0, seed=2020):
    print('Start processing!')
    train_dir1 = os.path.join(file_path, 'Training_Batch1')
    train_dir2 = os.path.join(file_path, 'Training_Batch2')
    filelist = glob.glob(os.path.join(train_dir1, 'volume-*.nii')) + glob.glob(os.path.join(train_dir2, 'volume-*.nii'))

    r = random.random
    random.seed(seed)
    random.shuffle(filelist, random=r)

    if train_size + val_size + test_size != 1:
        raise ValueError('The sum of train_size, val_size and test_size must be 1 !!!')
    
    N = len(filelist)
    train_size = round(N * train_size)
    val_size = round(N * val_size)
    test_size = round(N * test_size)

    liver_info = []
    tumor_info = []

    for i in range(N):
        image = nii_to_npy(filelist[i])
        label = nii_to_npy(filelist[i].replace('volume', 'segmentation'))
        imagename = os.path.splitext(os.path.split(filelist[i])[1])[0]
        labelname = imagename.replace('volume', 'segmentation')

        if i < train_size:
            current_path = os.path.join(save_path, 'train')
        elif train_size <= i < (train_size + val_size):
            current_path = os.path.join(save_path, 'val')
        else:
            current_path = os.path.join(save_path, 'test')

        if not os.path.exists(current_path):
            os.makedirs(current_path)

        # Save image and label
        np.save(os.path.join(current_path, imagename + '.npy'), image.astype(np.float16))
        np.save(os.path.join(current_path, labelname + '.npy'), label.astype(np.float16))

        # Location information of the liver
        if np.sum(label > 0):
            liver_xx, liver_yy, liver_zz = np.where(label > 0)
            liver_range_x = [np.min(liver_xx), np.max(liver_xx)]
            liver_range_y = [np.min(liver_yy), np.max(liver_yy)]
            liver_range_z = [np.min(liver_zz), np.max(liver_zz)]
            liver_range = liver_range_x + liver_range_y + liver_range_z
            liver_info.append(imagename)
        else:
            liver_range = [None, None, None, None, None, None]

        # Location information of the tumor
        if np.sum(label == 2):
            tumor_xx, tumor_yy, tumor_zz = np.where(label == 2)
            tumor_range_x = [np.min(tumor_xx), np.max(tumor_xx)]
            tumor_range_y = [np.min(tumor_yy), np.max(tumor_yy)]
            tumor_range_z = [np.min(tumor_zz), np.max(tumor_zz)]
            tumor_range = tumor_range_x + tumor_range_y + tumor_range_z
            tumor_info.append(imagename)
        else:
            tumor_range = [None, None, None, None, None, None]
        
        # Print the information
        liver_range_str = [str(i) for i in liver_range]
        liver_range_str = ' '.join(liver_range_str)
        tumor_range_str = [str(i) for i in tumor_range]
        tumor_range_str = ' '.join(tumor_range_str)
        current_info = imagename + ' liver_range ' + liver_range_str + ' tumor_range ' + tumor_range_str
        print(current_info)

        # Record the location information of the liver and tumor
        with open(os.path.join(current_path, 'data_info.txt'), 'a') as f:
            f.write(current_info + '\n')
            # e.g. 'volumn-0 liver_range 68 79 12 69 56 98 tumor_range 69 71 35 56 72 99'

    with open(os.path.join(save_path, 'whole_data_info.txt'), 'w') as f:
        liver_info_str = ' '.join(liver_info)
        tumor_info_str = ' '.join(tumor_info)
        f.write(liver_info_str + '\n')
        f.write(tumor_info_str)
    
    print('\nEnd of processing!')
    print('Total image:', N)
    print('The number of image with liver:', len(liver_info))
    print('The number of image with tumor:', len(tumor_info))

if __name__ == "__main__":
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    processing_data(file_path=file_path, train_size=0.8, val_size=0.2, test_size=0, seed=2020)
