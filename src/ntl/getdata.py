import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from scipy.special import softmax
from utils import get_vis_data, rgb_loader


IMAGE_SIZE = 64#32

dataTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
])

dataTransform_aug = transforms.Compose([
    transforms.ToTensor()
])
    
class Cus_Dataset(data.Dataset):
    def __init__(self, mode = None, \
                            dataset_1 = None, begin_ind1 = 0, size1 = 0,\
                            dataset_2 = None, begin_ind2 = 0, size2 = 0,\
                            dataset_3 = None, begin_ind3 = 0, size3 = 0,\
                            dataset_4 = None, begin_ind4 = 0, size4 = 0,\
                            new_model = None,
                            is_img_path = False, is_img_path_aug = [None, None, None, None]):

        self.mode = mode
        self.list_img = []
        self.list_img1 = []
        self.list_img2 = []
        
        self.list_label = []
        self.list_label1 = []
        self.list_label2 = []

        self.data_size = 0
        self.transform = dataTransform
        self.is_img_path = is_img_path
        self.is_img_path_aug = is_img_path_aug

        if self.mode == 'train_annotator': 
            
            self.data_size = size1

            path_list = dataset_1[0][begin_ind1: begin_ind1+size1]

            if self.is_img_path:
                for file_path in path_list:
                    img = np.array(rgb_loader(file_path))
                    self.list_img.append(img)
            else:
                for file_path in path_list:
                    self.list_img.append(file_path)


            self.list_label = dataset_1[1][begin_ind1: begin_ind1+size1]

            ind = np.arange(self.data_size)
            ind = np.random.permutation(ind)
            self.list_img = np.asarray(self.list_img)
            self.list_img = self.list_img[ind]

            self.list_label = np.asarray(self.list_label)
            self.list_label = self.list_label[ind]

        if self.mode == 'combined_training': 
            
            self.data_size = size1

            path_list1 = dataset_1[0][begin_ind1: begin_ind1+size1]
            path_list2 = dataset_2[0][begin_ind2: begin_ind2+size2]

            if self.is_img_path_aug[0]:
                for file_path in path_list1:
                    img = np.array(rgb_loader(file_path))
                    self.list_img1.append(img)
            else:
                for file_path in path_list1:
                    self.list_img1.append(file_path)

            if self.is_img_path_aug[1]:
                for file_path in path_list2:
                    img = np.array(rgb_loader(file_path))
                    self.list_img2.append(img)
            else:
                for file_path in path_list2:
                    self.list_img2.append(file_path)


            self.list_label1 = dataset_1[1][begin_ind1: begin_ind1+size1]
            self.list_label2 = dataset_2[1][begin_ind2: begin_ind2+size2]

            ind = np.arange(self.data_size)
            ind = np.random.permutation(ind)
            self.list_img1 = np.asarray(self.list_img1)
            self.list_img1 = self.list_img1[ind]
            self.list_img2 = np.asarray(self.list_img2)
            self.list_img2 = self.list_img2[ind]

            self.list_label1 = np.asarray(self.list_label1)
            self.list_label1 = self.list_label1[ind]
            self.list_label2 = np.asarray(self.list_label2)
            self.list_label2 = self.list_label2[ind]

        if self.mode == 'authorized_training': 
            
            self.data_size = size1

            path_list1 = dataset_1[0][begin_ind1: begin_ind1+size1]
            path_list2 = dataset_2[0][begin_ind2: begin_ind2+size2]
            path_list3 = dataset_3[0][begin_ind3: begin_ind3+size3]
            path_list4 = dataset_4[0][begin_ind4: begin_ind4+size4]
            label_list1 = dataset_1[1][begin_ind1: begin_ind1+size1]
            label_list2 = dataset_2[1][begin_ind2: begin_ind2+size2]
            label_list3 = dataset_3[1][begin_ind3: begin_ind3+size3]
            label_list4 = dataset_4[1][begin_ind4: begin_ind4+size4]

            for i in range(size1):
                self.list_img1.append(path_list1[i])
                self.list_label1.append(label_list1[i])

            for i in range(size2):
                self.list_img2.append(path_list2[i])
                self.list_label2.append(label_list2[i])

            for i in range(size3):
                self.list_img2.append(path_list3[i])
                self.list_label2.append(label_list3[i])

            for i in range(size4):
                self.list_img2.append(path_list4[i])
                self.list_label2.append(label_list4[i])


            ind = np.arange(self.data_size)
            ind = np.random.permutation(ind)
            self.list_img1 = np.asarray(self.list_img1)
            self.list_img1 = self.list_img1[ind]
            self.list_img2 = np.asarray(self.list_img2)
            self.list_img2 = self.list_img2[ind]

            self.list_label1 = np.asarray(self.list_label1)
            self.list_label1 = self.list_label1[ind]
            self.list_label2 = np.asarray(self.list_label2)
            self.list_label2 = self.list_label2[ind]


        elif self.mode == 'val': #val data

            self.data_size = size1
            path_list = dataset_1[0][begin_ind1: begin_ind1+size1]
            if self.is_img_path:
                for file_path in path_list:
                    img = np.array(rgb_loader(file_path))
                    self.list_img.append(img)
            else:
                for file_path in path_list:
                    self.list_img.append(file_path)

            self.list_label = dataset_1[1][begin_ind1: begin_ind1+size1]


    def __getitem__(self, item):
        if self.mode == 'train_annotator':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor(label)
        elif self.mode == 'combined_training':
            img1 = self.list_img1[item]
            img2 = self.list_img2[item]
            label1 = self.list_label1[item]
            label2 = self.list_label2[item]
            return self.transform(img1), torch.LongTensor(label1), self.transform(img2), torch.LongTensor(label2)
        elif self.mode == 'authorized_training':
            img1 = self.list_img1[item]
            img2 = self.list_img2[item]
            label1 = self.list_label1[item]
            label2 = self.list_label2[item]
            return self.transform(img1), torch.LongTensor(label1), self.transform(img2), torch.LongTensor(label2)
        elif self.mode == 'val':
            #img = Image.open(self.list_img[item])
            img = self.list_img[item]
            label = self.list_label[item]
            return self.transform(img), torch.LongTensor([label])


    def __len__(self):
        return self.data_size

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
