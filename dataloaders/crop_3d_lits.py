import os
import torch
import numpy as np
import glob

"""
patch size: (96, 128, 160)
media patient size (147, 216, 243)
"""
base_dir = 'E:/dataset/medical_data/LITS_Data/lits_npy'
# save_dir = 'E:/dataset/medical_data/LITS_Data/tumor_crop_data'
# save_dir = 'E:/dataset/medical_data/LITS_Data/margin_crop_data'
save_dir = 'E:/dataset/medical_data/LITS_Data/fixed_crop_data'

# def FixSizeCrop(image, label, crop_size=[112, 112, 80]):
#     crop_center
#     return image_crop, label_crop

def LiverCrop(image, label, liver_range, margin=15):
    liver_range = [int(i) for i in liver_range]

    crop_range_x = [max(liver_range[0] - margin, 0), min(liver_range[1] + margin, label.shape[0])]
    crop_range_y = [max(liver_range[2] - margin, 0), min(liver_range[3] + margin, label.shape[1])]
    crop_range_z = [max(liver_range[4] - margin, 0), min(liver_range[5] + margin, label.shape[2])]

    # Crop
    image_liver_crop = image[crop_range_x[0]:crop_range_x[1], crop_range_y[0]:crop_range_y[1], crop_range_z[0]:crop_range_z[1]]
    label_liver_crop = label[crop_range_x[0]:crop_range_x[1], crop_range_y[0]:crop_range_y[1], crop_range_z[0]:crop_range_z[1]]

    return image_liver_crop, label_liver_crop

def TumorCrop(image, label, tumor_range, margin=15):
    tumor_range = [int(i) for i in tumor_range]

    crop_range_x = [max(tumor_range[0] - margin, 0), min(tumor_range[1] + margin, label.shape[0])]
    crop_range_y = [max(tumor_range[2] - margin, 0), min(tumor_range[3] + margin, label.shape[1])]
    crop_range_z = [max(tumor_range[4] - margin, 0), min(tumor_range[5] + margin, label.shape[2])]

    # Crop
    image_tumor_crop = image[crop_range_x[0]:crop_range_x[1], crop_range_y[0]:crop_range_y[1], crop_range_z[0]:crop_range_z[1]]
    label_tumor_crop = label[crop_range_x[0]:crop_range_x[1], crop_range_y[0]:crop_range_y[1], crop_range_z[0]:crop_range_z[1]]

    return image_tumor_crop, label_tumor_crop

def FixedCrop(image, label, liver_range, crop_size=[384, 384, 384]):
    h, w, d = image.shape
    centerx, centery, centerz = (liver_range[1]+liver_range[0])//2, (liver_range[3]+liver_range[2])//2, (liver_range[5]+liver_range[4])//2
    croprangex = [centerx - crop_size[0]//2, centerx + crop_size[0]-crop_size[0]//2]
    croprangey = [centery - crop_size[1]//2, centery + crop_size[1]-crop_size[1]//2]
    croprangez = [centerz - crop_size[2]//2, centerz + crop_size[2]-crop_size[2]//2]
    # crop
    image = image[max(croprangex[0], 0):min(croprangex[1], h), max(croprangey[0], 0):min(croprangey[1], w), max(croprangez[0], 0):min(croprangez[1], d)]
    label = label[max(croprangex[0], 0):min(croprangex[1], h), max(croprangey[0], 0):min(croprangey[1], w), max(croprangez[0], 0):min(croprangez[1], d)]
    # padding
    padx0, padx1, pady0, pady1, padz0, padz1 = 0, 0, 0, 0, 0, 0
    if croprangex[0] < 0: padx0 = -croprangex[0]
    if croprangey[0] < 0: pady0 = -croprangey[0]
    if croprangez[0] < 0: padz0 = -croprangez[0]
    if croprangex[1] > h: padx1 = croprangex[1] - h
    if croprangey[1] > w: pady1 = croprangey[1] - w
    if croprangez[1] > d: padz1 = croprangez[1] - d
    image = np.pad(image, ((padx0, padx1), (pady0, pady1), (padz0, padz1)), 'constant', constant_values=0)
    label = np.pad(label, ((padx0, padx1), (pady0, pady1), (padz0, padz1)), 'constant', constant_values=0)
    return image, label

def Crop_3d_lits(file_path, save_path):
    # Read the location information about  livers and tumors
    # e.g. 'volumn-0 liver_range 68 79 12 69 56 98 tumor_range 69 71 35 56 72 99'
    data_info_path = os.path.join(file_path, 'data_info.txt')
    with open(data_info_path, 'r') as f:
        data_info = f.read()
    data_info = data_info.split('\n')
    data_info = [i.split(' ') for i in data_info]
    image_list = [os.path.join(file_path, data_info[i][0] + '.npy') for i in range(len(data_info))]

    print("total {} samples".format(len(image_list)))

    for idx in range(len(image_list)):
        image_name = image_list[idx]
        image = np.load(image_name)
        label = np.load(image_name.replace('volume', 'segmentation'))

        liver = np.where(label==1)
        liver_range = [liver[0].min(), liver[0].max(), liver[1].min(), liver[1].max(), liver[2].min(), liver[2].max()]
        # tumor = np.where(label==2)
        # tumor_range = [tumor[0].min(), tumor[0].max(), tumor[1].min(), tumor[1].max(), tumor[2].min(), tumor[2].max()]

        # if tumor_range[0] == 'None':
        #     continue
        # image_liver_crop, label_liver_crop = LiverCrop(image, label, liver_range=liver_range, margin=10)
        # image_liver_crop, label_liver_crop = TumorCrop(image, label, tumor_range=tumor_range, margin=10)
        image_liver_crop, label_liver_crop = FixedCrop(image, label, liver_range, crop_size=[384, 384, 384])
        if image.shape != label.shape:
            print('The shape of this image and label are different!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(image_name)
            print('Image shape', image.shape)
            print('Label shpae', label.shape)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        np.save(os.path.join(save_path, data_info[idx][0] + '.npy'), image_liver_crop.astype(np.float16))
        np.save(os.path.join(save_path, data_info[idx][0].replace('volume', 'segmentation') + '.npy'), label_liver_crop.astype(np.float16))
        print(data_info[idx][0], '\tBefore crop', image.shape, '\tAfter crop', image_liver_crop.shape, '\tLiver range:', liver_range)

        # Print the information
        # image_size_str = [str(i) for i in image.shape]
        # image_size_str = ' '.join(image_size_str)
        # tumor_range_str = [str(i) for i in tumor_range[idx]]
        # tumor_range_str = ' '.join(tumor_range_str)
        # current_info = data_info[idx][0] + ' image_size ' + image_size_str + ' tumor_range ' + tumor_range_str
        current_info = data_info[idx][0]

        # Record the location information of the liver and tumor
        with open(os.path.join(save_path, 'data_info.txt'), 'a') as f:
            f.write(current_info + '\n')
            # e.g. 'volumn-0 image_size 68 79 12 69 56 98 tumor_range 69 71 35 56 72 99'

if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for split in ['val', 'train']:
        _base_dir = os.path.join(base_dir, split)
        _save_dir = os.path.join(save_dir, split)
        if not os.path.exists(_save_dir):
            os.makedirs(_save_dir)
        
        if os.path.exists(os.path.join(_base_dir, 'data_info.txt')):
            print('Processing begin--' + split)
            Crop_3d_lits(_base_dir, _save_dir)