import os
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from Augmentation import *

IM_SIZE = 101

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

def normalize(im):
        max_value = np.max(im)
        min_value = np.min(im)
        if max_value > min_value:
            im = (im - min_value) / (max_value - min_value)
        elif max_value == min_value:
            im = im / 255.0
        return im

        
def basic_augment(image, mask, index):    
    # one half chance to do the operation of flipping
    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip(image, mask)
        pass
    
    if np.random.rand() < 0.5:
        # one quarter chance to do one of these operations on the size of image and mask
        c = np.random.choice(4)
        if c == 0:
            image, mask = do_random_shift_scale_crop_pad(image, mask, 0.2)
        elif c == 1:
            image, mask = do_horizontal_shear(image, mask, dx=np.random.uniform(-0.07, 0.07))
        elif c == 2:
            image, mask = do_shift_scale_rotate(image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 15))
        elif c == 3:
            image, mask = do_elastic_transform(image, mask, grid=10, distort=np.random.uniform(0, 0.15))
            
    if np.random.rand() < 0.5:      
        # one third chance to do one of these operations on the brightness of image
        c = np.random.choice(3)
        if c == 0:
            image = do_brightness_shift(image, np.random.uniform(-0.1, 0.1))
        elif c == 1:
            image = do_brightness_multiply(image, np.random.uniform(1 - 0.08, 1 + 0.08))
        elif c == 2:
            image = do_gamma(image, np.random.uniform(1 - 0.08, 1 + 0.08))
            
    return image, mask, index

def do_length_decode(rle, H, W, fill_value=255):
    mask = np.zeros((H, W), np.uint8)
    if rle == '' or rle == 'nan':
        return mask
    else:
        mask = mask.reshape(-1)
        rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
        for r in rle:
            start = r[0] - 1
            end = start + r[1]
            mask[start:end] = fill_value
        
        mask = mask.reshape(W, H).T
        return mask
    
def modify_empty_label_with_pseudo_label(df, folder_path='./Saves/Res34Unetv4/Res34Unetv4_5foldAvg.csv'):
    df_actlabel = df[df.rle_mask < 'nan']
    df_fakelabel = df.drop(df[df.rle_mask < 'nan'].index)
    df_pseudo = pd.read_csv(folder_path)
    df_pseudo['masks'] = [do_length_decode(str(p), 101, 101) for p in df_pseudo.rle_mask]
    # modify fake label with pseudo label
    df_fakelabel.rle_mask = df_pseudo.rle_mask
    df_fakelabel.masks = df_pseudo.masks
    # concate true label and fake label together
    dff = df_actlabel.append(df_fakelabel)
    dff = dff.sort_values('id')
    dff['masks'] = [np.array(normalize(dff.iloc[i]['masks'])).astype(np.float32) for i in range(len(dff))]
    
    for i in range(len(dff)):
        if str(dff.rle_mask.iloc[i]) == 'nan':
            dff.rle_mask.iloc[i] = float(dff.rle_mask.iloc[i])
        if str(dff.masks.iloc[i]) == 'nan':
            dff.masks.iloc[i] = np.zeros((101, 101), np.float32)
    return dff
    
class TGS_Dataset(Dataset):
    def __init__(self, folder_path='./Data/train', modify=False):
        self.folder_path = folder_path
        if modify:
            self.df = self.create_dataset_df(self.folder_path, modify=True)
        else:
            self.df = self.create_dataset_df(self.folder_path)
        self.df['z'] = normalize(self.df['z'].values)
        if self.folder_path == './Data/train':
            self.df_actual = self.df[self.df.rle_mask < 'nan']
            self.df_pseudo = self.df.drop(self.df[self.df.rle_mask < 'nan'].index)
            try:
                empty = np.array([np.sum(m) for m in self.df['masks']])
                print('{} empty masks out of {} total masks'.format((np.sum(empty == 0)), len(empty)))
            except AttributeError:
                pass
                    
    @staticmethod
    def create_dataset_df(folder_path, load = True, modify=False):
        '''Create a dataset for a specific dataset folder path '''
        if folder_path == './Data/train':
            walk = os.walk(folder_path)
            main_dir_path, subdirs_path, csv_path = next(walk)
            dir_img_path, _, img_path = next(walk)
        elif folder_path == './Data/test':
            walk = os.walk(folder_path)
            main_dir_path, subdirs_path, csv_path = next(walk)
#             check_dir_path, _, _ = next(walk)
            dir_img_path, _, img_path = next(walk)
        # create dataframe
        df = pd.DataFrame()
        df['id'] = [img_p.split('.')[0] for img_p in img_path]
        df['img_path'] = [os.path.join(dir_img_path, img_p) for img_p in img_path]
        # if the folder is training folder, which includes mask folder, do the following code
        if any(['mask' in sub for sub in subdirs_path]):
            data = 'train'
            dir_mask_path, _, mask_path = next(walk)
            df['mask_path'] = [os.path.join(dir_mask_path, mask_p) for mask_p in mask_path]
            # read in training.csv file
            rle_df = pd.read_csv(os.path.join(main_dir_path, csv_path[1]))
            df = df.merge(rle_df, on='id', how='left')    
        # if the folder is testing folder, ignore it
        else:
            data = 'test'
        # read in csv file
        depth_df = pd.read_csv(os.path.join(main_dir_path, csv_path[0]))
        df = df.merge(depth_df, on='id', how='left')
        if load:
            df = TGS_Dataset.load_images(df, data = data)
        if modify:
            df = modify_empty_label_with_pseudo_label(df)
        return df        
    
    @staticmethod
    def load_images(df, data='train'):
        df['images'] = [
            normalize(cv2.imread(df.iloc[i]['img_path'], cv2.IMREAD_COLOR).astype(np.float32)) for i in range(len(df))]
        if data == 'train':
            df['masks'] = [
                normalize(cv2.imread(df.iloc[i]['mask_path'], cv2.IMREAD_GRAYSCALE).astype(np.float32)) for i in range(len(df))]
        return df
    
    # self define a dataloader which modifies a little bit based on DataLoader
    def selfdefine_dataloader_actual(self, flag=False, nfold=5, shuffle=True, seed=143, num_workers=8, batch_size=10):
        if not flag:
            kf = KFold(n_splits=nfold, shuffle=True, random_state=seed)
            self.df_actual['coverage'] = self.df_actual.masks.map(np.sum) / pow(IM_SIZE, 2)
            self.df_actual['coverage_class'] = self.df_actual.coverage.map(cov_to_class)
            
            loaders = []
            idx = []
            for train_ids, val_ids in kf.split(self.df_actual['id'].values, self.df_actual.coverage_class):
                train_df = self.df_actual.iloc[train_ids]
                train_dataset = TorchDataset(train_df, transform=basic_augment)
                train_loader = DataLoader(train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
                
                val_df = self.df_actual.iloc[val_ids]
                val_dataset = TorchDataset(val_df, transform=None)
                val_loader = DataLoader(val_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
                
                idx.append((self.df_actual.id.iloc[train_ids], self.df_actual.id.iloc[val_ids]))
                loaders.append((train_loader, val_loader))
                
            return loaders, idx
        
        else:        
            pseudo_dataset = TorchDataset(self.df_pseudo, is_test=True)
            pseudo_loader = DataLoader(pseudo_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)
            
            return pseudo_loader, self.df_pseudo.id             
            
         
    def selfdefine_dataloader(self, data='train', nfold=5, shuffle=True, seed=143, num_workers=8, batch_size=10):
        if data == 'train':
            kf = KFold(n_splits=nfold, shuffle=True, random_state=seed)
            self.df['coverage'] = self.df.masks.map(np.sum) / pow(IM_SIZE, 2)
            self.df['coverage_class'] = self.df.coverage.map(cov_to_class)
                
            loaders = []
            idx = []
            for train_ids, val_ids in kf.split(self.df['id'].values, self.df.coverage_class):
                train_df = self.df.iloc[train_ids]
                train_dataset = TorchDataset(train_df, transform=basic_augment)
                train_loader = DataLoader(train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
                
                val_df = self.df.iloc[val_ids]
                val_dataset = TorchDataset(val_df, transform=None)
                val_loader = DataLoader(val_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
                
                idx.append((self.df.id.iloc[train_ids], self.df.id.iloc[val_ids]))
                loaders.append((train_loader, val_loader))
                
            return loaders, idx        
              
        elif data == 'test':        
            test_dataset = TorchDataset(self.df, is_test=True)
            test_loader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size)
            
            return test_loader, self.df.id
            
                                                                               
class TorchDataset(Dataset):
    def __init__(self, df, is_test=False, transform=None):
        self.df = df
        self.is_test = is_test
        self.transform = transform
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        # define an array to fill the edge of the original mask array
        pad = ((0, 0), (14, 13), (14, 13))
        im = self.df.images.iloc[index]
        # if the data is from training folder
        if not self.is_test:
            # deal with mask
            mask = self.df.masks.iloc[index]
            # if transform is needed
            if self.transform is not None:
                im, mask, index = self.transform(im, mask, index)
            # add one dimension at the first dimension position
            mask = np.expand_dims(mask, 0)
            mask = np.pad(mask, pad, 'edge')
            # transform from numpy to tensor
            mask = torch.from_numpy(mask).float()
            
        # deal with image            
        im = np.rollaxis(im, 2, 0)
        im = np.pad(im, pad, 'edge')
        im = torch.from_numpy(im).float()
        z = torch.from_numpy(np.expand_dims(self.df.z.iloc[index], 0)).float()
        
        if self.is_test:
            return self.df.id.iloc[index], im, z
        else:
            return self.df.id.iloc[index], im, mask, z
        
        
        
            
            
            
            
                
            
        
    
    
    
        

