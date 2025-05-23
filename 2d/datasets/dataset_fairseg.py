import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom

from torch.utils.data import Dataset
from einops import repeat, rearrange
from icecream import ic

import pandas as pd
from torchvision import transforms
import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from icecream import ic
from PIL import Image

#########


hashmap = {-1:1, -2:2, 0:0}

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    if len(image.shape) > 2:
        image = np.stack([np.rot90(image[..., idx], k) for idx in range(image.shape[-1])], axis=-1)
    else:
        image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    if len(image.shape) > 2:
        image = np.stack([np.flip(image[..., idx], axis=axis) for idx in range(image.shape[-1])], axis=-1)
    else:
        image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    if len(image.shape) > 2:
        image = np.stack([ndimage.rotate(image[..., idx], angle, order=0, reshape=False) for idx in range(image.shape[-1])], axis=-1)
    else:
        image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def crop_center(img, cropx, cropy):
    
    if len(img.shape) > 2:
        y,x,ch = img.shape
        startx = max(0, x//2-(cropx//2))
        starty = max(0, y//2-(cropy//2))    
        endx = startx + min(cropx, x)
        endy = starty + min(cropy, y)
        buffer_startx = max(0, cropx // 2 - x // 2)
        buffer_starty = max(0, cropy // 2 - y // 2)
        buffer_endx = buffer_startx + (endx - startx)
        buffer_endy = buffer_starty + (endy - starty)

        buffer_img = img[
            starty:endy,
            startx:endx,
            ...
        ]
        return buffer_img
    else:
        y,x = img.shape
        startx = max(0, x//2-(cropx//2))
        starty = max(0, y//2-(cropy//2))    
        endx = startx + min(cropx, x)
        endy = starty + min(cropy, y) 
        buffer_startx = max(0, cropx // 2 - x // 2)
        buffer_starty = max(0, cropy // 2 - y // 2)
        buffer_endx = buffer_startx + (endx - startx)
        buffer_endy = buffer_starty + (endy - starty)
        
        buffer_img = img[
            starty:endy,
            startx:endx
        ]
    return buffer_img

class RandomGenerator(object):
    def __init__(self, output_size, center_crop_size, use_normalize=False):
        self.output_size = output_size
        self.a_min, self.a_max = 0, 255
        self.use_normalize = use_normalize
        self.center_crop_size = center_crop_size

    def __call__(self, sample):
        image, label, attr_label, pid = sample['image'], sample['label'], sample['attr_label'], sample['pid'] 

        # print(image.max())
        # print(image.min())
        image = np.clip(image, self.a_min, self.a_max)
        
        
        if self.use_normalize:
            assert self.a_min != self.a_max
            image = (image - self.a_min) / (self.a_max - self.a_min)     

        ## convert label to training format
        for k in sorted(hashmap.keys()):
            label[label == k] = hashmap[k]
            
        image = crop_center(image, self.center_crop_size, self.center_crop_size)
        label = crop_center(label, self.center_crop_size, self.center_crop_size)
                        
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
            
        x, y = label.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            if len(image.shape) > 2:
                image = np.stack([zoom(image[..., idx], (self.output_size[0] / x, self.output_size[1] / y), order=3) for idx in range(image.shape[-1])], axis=-1)
            else:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # # debug
        # image = image*255
        # image = image.astype(np.uint8)
        # im = Image.fromarray(image)
        # im = im.convert("RGB")
        # im.save("/home/local/PARTNERS/yo084/1_code/18.1_dMoE/2D/FairSeg/SAMed/datasets/image.jpg")
        # lab = Image.fromarray(label*50)
        # lab = lab.convert("L")
        # lab.save("/home/local/PARTNERS/yo084/1_code/18.1_dMoE/2D/FairSeg/SAMed/datasets/label.jpg")
        # exit()
        
        if len(image.shape) > 2:
            image = torch.from_numpy(image.astype(np.float32))
            image = rearrange(image, 'h w c -> c h w')
        else:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        attr_label = torch.tensor(attr_label).long()
        sample = {'image': image, 'label': label.long(), 'attr_label': attr_label, 'pid': pid}
        
        return sample



class TestGenerator(object):
    def __init__(self, output_size, center_crop_size, use_normalize=False):
        self.output_size = output_size
        self.a_min, self.a_max = 0, 255
        self.use_normalize = use_normalize
        self.center_crop_size = center_crop_size

    def __call__(self, sample):
        image, label, attr_label, pid = sample['image'], sample['label'], sample['attr_label'], sample['pid'] 

        image = np.clip(image, self.a_min, self.a_max)
        if self.use_normalize:
            assert self.a_min != self.a_max
            image = (image - self.a_min) / (self.a_max - self.a_min)     

        ## convert label to training format
        for k in sorted(hashmap.keys()):
            label[label == k] = hashmap[k]
        
        image = crop_center(image, self.center_crop_size, self.center_crop_size)
        label = crop_center(label, self.center_crop_size, self.center_crop_size)
      
        x, y = label.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            if len(image.shape) > 2:
                image = np.stack([zoom(image[..., idx], (self.output_size[0] / x, self.output_size[1] / y), order=3) for idx in range(image.shape[-1])], axis=-1)
            else:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        if len(image.shape) > 2:
            image = torch.from_numpy(image.astype(np.float32))
            image = rearrange(image, 'h w c -> c h w')
        else:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        attr_label = torch.tensor(attr_label).long()

        sample = {'image': image, 'label': label.long(), 'attr_label': attr_label, 'pid': pid}
        return sample



# attr_to_race = {2: 0, 1: 1, 0: 2}
attr_to_language = {0: 0, 1: 1, 2:2, -1:-1}



class FairSeg_dataset(Dataset):
    def __init__(self, base_dir, split, args, balanced=False, bal_attr='race', \
                 resolution=224, transform=None, attr_label='race', img_type='slo_fundus'):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir
        self.args = args
        self.img_type = img_type
        self.needBalance = balanced
        
        # Assign a unique number to each label
        labels = ["scalp", "face", "back", "trunk", "chest", "upper extremity", "abdomen", "lower extremity", "neck", "foot"]
        label_dict = {label: idx for idx, label in enumerate(labels, start=1)}
        def assign_number(input_text):
            return label_dict.get(input_text, -1)
        
        list_dir = args.list_dir
        if list_dir.find('HAM') >= 0:
            self.load_jpg = True
            self.sample_list = []
            sample_list = pd.read_csv(list_dir+'/'+self.split+'.csv')
            unit_no = list(sample_list['image_id'])
            for no in unit_no:
                d = {}
                d['pid'] = no
                row = sample_list.loc[sample_list['image_id'] == no]
                d['image_path'] = row["Path"].values[0]
                d['label_path'] = d['image_path'].replace('_images', '_segmentations').replace('.jpg', '_segmentation.png')
                # d['attr_label'] = 1 if row["Sex"].values[0] == 'M' else 0
                if attr_label == 'age':
                    d['attr_label'] = row["Age_multi"].values[0]
                elif attr_label == 'gender':
                    d['attr_label'] = 1 if row["Sex"].values[0] == 'M' else 0
                elif attr_label == 'local':
                    try:
                        d['attr_label'] = int(assign_number(row["localization"].values[0]))-1
                    except:
                        continue
                    if d['attr_label'] < 0:
                        continue
                    # print(d['attr_label'])
                self.sample_list.append(d)
                # print(d)
        else:
            self.load_jpg = False
            self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
            self.needBalance = True
        
        self.attr_label = attr_label
        print(attr_label)
        
        self.bal_attr = bal_attr
        self.balance_factor = 1.
        self.label_samples = dict()
        self.class_samples_num = None
        self.balanced_max = 0
        self.per_attr_samples = dict()
        
        # all_files = self.find_all_files(self.data_dir, suffix='npz')
        
        self.resolution = resolution
        if self.attr_label == 'race' or self.attr_label == 'language': 
            self.sens_classes = 3
        else:
            self.sens_classes = 2

        if self.split == 'train' and self.needBalance:
            self.screw = args.screw
            if args.screw < 4:
                self.data_files = self.bal_samples_based_attr(self.sample_list)
            else:
                self.data_files = self.sample_list
            # self.data_files = self.bal_samples_based_attr(self.sample_list)
        else: # testing set
            # self.data_files = all_files[8000:]
            self.data_files = self.sample_list

        if self.attr_label == 'language':
            self.data_files_ = []
            for idx in range(0, len(self.data_files)):
                data_path = os.path.join(self.data_dir, self.data_files[idx].strip('\n'))
                data = np.load(data_path, allow_pickle=True)
                attr_label = data[self.attr_label].item()
                # print(attr_label)
                if attr_label >= 0:
                    self.data_files_.append(self.data_files[idx])
            self.data_files = self.data_files_

    def __len__(self):
        return len(self.data_files)
    
    # def bal_samples_based_attr(self, all_files):
        
    #     for idx in range(0, len(all_files)):
    #         npz_file = os.path.join(self.data_dir, all_files[idx])
    #         raw_data = np.load(npz_file, allow_pickle=True)
    #         cur_attr_label = raw_data[self.bal_attr].item() # self.race_mapping[raw_data['race'].item()]
    #         if cur_attr_label not in self.per_attr_samples:
    #             self.per_attr_samples[cur_attr_label] = list()
    #         self.per_attr_samples[cur_attr_label].append(all_files[idx])
    #         self.balanced_max = len(self.per_attr_samples[cur_attr_label]) \
    #             if len(self.per_attr_samples[cur_attr_label]) > self.balanced_max else self.balanced_max
    #     ttl_num_samples = 0
    #     self.class_samples_num = [0]*len(list(self.label_samples.keys()))
    #     for i, (k,v) in enumerate(self.label_samples.items()):
    #         self.class_samples_num[int(k)] = len(v)
    #         ttl_num_samples += len(v)
    #         print(f'{k}-th identity training samples: {len(v)}')
    #     print(f'total number of training samples: {ttl_num_samples}')
    #     self.class_samples_num = np.array(self.class_samples_num)

    #     # Oversample the classes with fewer elements than the max
    #     for i_label in self.label_samples:
    #         while len(self.label_samples[i_label]) < self.balanced_max*self.balance_factor:
    #             self.label_samples[i_label].append(random.choice(self.label_samples[i_label]))
        
    #     data_files = []
    #     for i, (k,v) in enumerate(self.label_samples.items()):
    #         data_files = data_files + v
        
    #     return data_files

    def bal_samples_based_attr(self, all_files):
        
        for idx in range(0, len(all_files)):
            print(f'{idx} : {all_files[idx]}')
            npz_file = os.path.join(self.data_dir, all_files[idx].strip('\n'))
            raw_data = np.load(npz_file, allow_pickle=True)
            cur_attr_label = raw_data[self.bal_attr].item() # self.race_mapping[raw_data['race'].item()]
            if cur_attr_label not in self.per_attr_samples:
                self.per_attr_samples[cur_attr_label] = list()
            self.per_attr_samples[cur_attr_label].append(all_files[idx].strip('\n'))
            self.balanced_max = len(self.per_attr_samples[cur_attr_label]) \
                if len(self.per_attr_samples[cur_attr_label]) > self.balanced_max else self.balanced_max
        
        # print(self.per_attr_samples)
        num_attr_list = []
        for attr in range(3):
            num_attr = len(self.per_attr_samples[attr])
            print(f'{attr} : {num_attr}')   
            num_attr_list.append(num_attr)
            
        for attr in range(3):   
            if attr == self.screw:
                self.per_attr_samples[attr] = self.per_attr_samples[attr][:750]
            # else:
            #     self.per_attr_samples[attr] = self.per_attr_samples[attr][:100]
            num_attr = len(self.per_attr_samples[attr])
            print(f'{attr} : {num_attr}') 
            
        # ttl_num_samples = 0
        # self.class_samples_num = [0]*len(list(self.label_samples.keys()))
        # for i, (k,v) in enumerate(self.label_samples.items()):
        #     self.class_samples_num[int(k)] = len(v)
        #     ttl_num_samples += len(v)
        #     print(f'{k}-th identity training samples: {len(v)}')
        # print(f'total number of training samples: {ttl_num_samples}')
        # self.class_samples_num = np.array(self.class_samples_num)

        # # Oversample the classes with fewer elements than the max
        # for i_label in self.label_samples:
        #     while len(self.label_samples[i_label]) < self.balanced_max*self.balance_factor:
        #         self.label_samples[i_label].append(random.choice(self.label_samples[i_label]))
        
        data_files = []
        for i, (k,v) in enumerate(self.per_attr_samples.items()):
            data_files = data_files + v
        
        total_len = len(data_files)
        print(f'total : {total_len}')   
        
        return data_files
    

    def group_counts(self, resample_which = 'group'):
        
        # if self.sens_name == 'Sex':
        #     mapping = {'M': 0, 'F': 1}
        #     groups = self.dataframe['Sex'].values
        #     group_array = [*map(mapping.get, groups)]
            
        # elif self.sens_name == 'Age':
        #     if self.sens_classes == 2:
        #         groups = self.dataframe['Age_binary'].values
        #     elif self.sens_classes == 5:
        #         groups = self.dataframe['Age_multi'].values
        #     elif self.sens_classes == 4:
        #         groups = self.dataframe['Age_multi4'].values.astype('int')
        #     group_array = groups.tolist()
            
        # elif self.sens_name == 'Race':
        #     mapping = {'White': 0, 'non-White': 1}
        #     groups = self.dataframe['Race'].values
        #     group_array = [*map(mapping.get, groups)]
        # elif self.sens_name == 'skin_type':
        #     if self.sens_classes == 2:
        #         groups = self.dataframe['skin_binary'].values
        #     elif self.sens_classes == 6:
        #         groups = self.dataframe['skin_type'].values
        #     group_array = groups.tolist()
        # elif self.sens_name == 'Insurance':
        #     if self.sens_classes == 2:
        #         groups = self.dataframe['Insurance_binary'].values
        #     elif self.sens_classes == 5:
        #         groups = self.dataframe['Insurance'].values
        #     group_array = groups.tolist()
        # else:
        #     raise ValueError("sensitive attribute does not defined in BaseDataset")
        
        # if resample_which == 'balanced':

        #     #get class
        #     labels = self.Y.tolist()
        #     num_labels = len(set(labels))
        #     num_groups = len(set(group_array))
            
        #     group_array = (np.asarray(group_array) * num_labels + np.asarray(labels)).tolist()

        group_count = [0] * self.sens_classes
       
        for idx in range(0, len(self.data_files)):
            npz_file = os.path.join(self.data_dir, self.data_files[idx].strip('\n'))
            raw_data = np.load(npz_file, allow_pickle=True)
            attr_label = raw_data[self.attr_label].item() 
            if self.attr_label == "age":
                attr_label=attr_label/365
                if attr_label < 60:
                    attr_label = 0
                else:
                    attr_label = 1
            elif self.attr_label == "maritalstatus":
                if attr_label != 0 and attr_label != -1: 
                    attr_label = 1
            elif self.attr_label == 'race':
                attr_label = attr_to_race[attr_label]
            elif self.attr_label == 'language':
                attr_label = attr_to_language[attr_label]
            if attr_label != -1:
                # print(attr_label)
                group_count[attr_label]+=1
        # self._group_array = torch.LongTensor(group_array)
        # if resample_which == 'group':
        #     self._group_counts = (torch.arange(self.sens_classes).unsqueeze(1)==self._group_array).sum(1).float()
        # elif resample_which == 'balanced':
        #     self._group_counts = (torch.arange(num_labels * num_groups).unsqueeze(1)==self._group_array).sum(1).float()
        # elif resample_which == 'class':
        #     self._group_counts = (torch.arange(num_labels).unsqueeze(1)==self._group_array).sum(1).float()
        self._group_counts = group_count
        self._group_counts = torch.tensor(self._group_counts).float()
        return self._group_counts

    def find_all_files(self, folder, suffix='npz'):
        files = [f for f in os.listdir(folder) \
                 if os.path.isfile(os.path.join(folder, f)) and \
                    os.path.join(folder, f).endswith(suffix)]
        return files

    def __getitem__(self, idx):

        if self.load_jpg:
            attr_label = self.data_files[idx]['attr_label']
            pid = self.data_files[idx]['pid']
            image = np.asarray(Image.open(os.path.join(self.data_dir, self.data_files[idx]['image_path'])))
            label = np.asarray(Image.open(os.path.join(self.data_dir, self.data_files[idx]['label_path'])), dtype=np.int8)
            label[label==255] = -1

        else:
            data_path = os.path.join(self.data_dir, self.data_files[idx].strip('\n'))
            
            data = np.load(data_path, allow_pickle=True)
            # print(list(data.keys()))
        
            image, label = data[self.img_type], data['disc_cup_mask']
            
            # print(self.attr_label)
            attr_label = data[self.attr_label].item()
            # for keys in data.keys():
            #     print(self.data_files[idx].strip('\n'))
            #     print(f'{keys} - {data[self.attr_label].item()}')

            # print(data['maritalstatus'].item())
        
            if self.attr_label == "age":
                attr_label=attr_label/365
                if attr_label < 60:
                    attr_label = 0
                else:
                    attr_label = 1
            elif self.attr_label == "maritalstatus":
                if attr_label != 0 and attr_label != -1: 
                    attr_label = 1
            elif self.attr_label == 'language':
                attr_label = attr_to_language[attr_label]
            # print(attr_label)

            # print(data['language'].item())
            # if language == 'English':
            # language_t = 0
            # elif language == 'Spanish':
            # language_t = 1
            # elif language == 'Unavailable' or language == 'NULL':
            # language_t = -1
            # else:
            # language_t = 2
            
            # pid = data['pid'].item()
            pid = self.data_files[idx].split('.npz')[0]
            # Input dim should be consistent
            # Since the channel dimension of nature image is 3, that of medical image should also be 3
            # ['fundus_slo', 'disc_cup', 'axes_cup', 'axes_disc', 'md', 'tds', \
            # 'pid', 'maritalstatus', 'hispanic', 'language', 'gender', 'race', 'age', 'datadir']

        sample = {'image': image, 'label': label, 'attr_label': attr_label, 'pid': pid}

        if self.transform:
            sample = self.transform(sample)
        
        return sample



