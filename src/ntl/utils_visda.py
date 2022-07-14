import numpy as np
from collections.abc import Iterable
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import math
from scipy.special import softmax
import scipy.io as sio
import torchvision.datasets as datasets
import cv2

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_visda_data(source):
    list_img = []
    list_label = []
    data_size = 0
    root_temp = "/home/code/nontrans/data/VisDA/{}".format(source)
    class_path = os.listdir(root_temp)
    for i in range(len(class_path)):
        class_temp = os.path.join(root_temp, class_path[i])
        img_path = os.listdir(class_temp)
        for j in range(1000):
            img_path_temp = os.path.join(class_temp, img_path[j])
            img = cv2.imread(img_path_temp)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112))
            
            list_img.append(img)
            list_label.append(np.eye(12)[i])
            data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]

def get_augment_data():
    list_img = []
    list_label = []
    data_size = 0

    augment_trainset = os.listdir("/home/GAN_aug/augment_visda_1/")
    augment_labels = np.loadtxt("/home/GAN_aug/augment_visda_1/labels")

    for i in range(len(augment_trainset) - 1):
        img = cv2.imread("/home/GAN_aug/augment_visda_1/img_{}.png".format(i))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        list_img.append(img)
        # print(int(augment_labels[i]))
        list_label.append(np.eye(12)[int(augment_labels[i])])
        data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]

    
def add_watermark(dataset_list, img_size = 112):
    list_img, list_label, data_size = dataset_list
    
    mask = np.zeros(list_img[0].shape)
    for i in range(img_size):
        for j in range(img_size):
            if i % 2 == 0 and j % 2 == 0:
                mask[i,j,0] = 100
    img_list_len = list_img.shape[0]
    #print(np.max(list_img[i]))
    for i in range(img_list_len):
        list_img[i] = np.minimum(list_img[i].astype(int) + mask.astype(int), 255).astype(np.uint8)
    return [list_img, list_label, data_size]


def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)
