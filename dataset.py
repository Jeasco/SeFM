import os
import cv2
import torch
import random
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from data_util import RandomCrop, RandomRotation, RandomResizedCrop, RandomHorizontallyFlip, RandomVerticallyFlip


class TrainDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.image_size = opt.image_size

        self.dataset = os.path.join(opt.train_dataset, 'train.txt')
        self.image_path = opt.train_dataset
        self.mat_files = open(self.dataset, 'r').readlines()
        self.file_num = len(self.mat_files)
        transform_list = []

        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        img_file = file_name.split(' ')[0]
        gt_file = file_name.split(' ')[1].strip()

        label = img_file.split('/')[-1][:-4]
        label = np.array([np.float32(i) for i in label])

        in_img = cv2.imread(self.image_path+img_file)
        gt_img = cv2.imread(self.image_path+gt_file)

        inp_img = Image.fromarray(in_img)
        tar_img = Image.fromarray(gt_img)

        inp_img = self.transform(inp_img)
        tar_img = self.transform(tar_img)

        sample = {'in_img': inp_img, 'gt_img': tar_img, 'label': label}
        return sample


class ValDataset(Dataset):
    def __init__(self, opt):
        super(ValDataset, self).__init__()

        self.dataset = os.path.join(opt.val_dataset, 'test.txt')
        self.image_path = opt.val_dataset

        self.mat_files = open(self.dataset, 'r').readlines()
        self.file_num = len(self.mat_files)

        self.ps = opt.image_size

        transform_list = []

        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        img_file = file_name.split(' ')[0]
        gt_file = file_name.split(' ')[1].strip()

        in_img = cv2.imread(self.image_path + img_file)
        gt_img = cv2.imread(self.image_path + gt_file)

        in_img = Image.fromarray(in_img)
        gt_img = Image.fromarray(gt_img)

        in_img = self.transform(in_img)
        gt_img = self.transform(gt_img)


        sample = {'in_img': in_img, 'gt_img': gt_img}
        return sample

class TestDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.dataset = os.path.join(opt.test_dataset, 'test.txt')
        self.image_path = opt.dataset_path

        self.mat_files = open(self.dataset, 'r').readlines()
        self.file_num = len(self.mat_files)
        transform_list = []

        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        img_file = file_name.split(' ')[0]
        gt_file = file_name.split(' ')[1].strip()

        in_img = cv2.imread(self.image_path + img_file)
        gt_img = cv2.imread(self.image_path + gt_file)

        in_img = Image.fromarray(in_img)
        gt_img = Image.fromarray(gt_img)

        in_img = self.transform(in_img)
        gt_img = self.transform(gt_img)

        img_path = img_file.split('/')
        sample = {'in_img': in_img, 'gt_img': gt_img, 'image_name':img_path[-1][:-4]}

        return sample

