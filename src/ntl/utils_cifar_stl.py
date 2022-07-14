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

def get_cifar_data():
    list_img = []
    list_label = []
    data_size = 0
    dir = '/home/GAN_aug/data/cifar/cifar-10-batches-py/'

    for filename in ['%s/data_batch_%d' % (dir,j) for j in range(1, 6)]:
        with open(filename, 'rb') as fo:
            cifar10 = pickle.load(fo, encoding = 'bytes')
        for i in range(len(cifar10[b"labels"])):
            img = np.reshape(cifar10[b"data"][i], (3,32,32))
            img = np.transpose(img, (1, 2, 0))
            #img = img.astype(float)
            list_img.append(img)
            
            list_label.append(np.eye(10)[cifar10[b"labels"][data_size%10000]])
            data_size += 1

    # np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]

    return [list_img, list_label, data_size]

def get_stl_data():
    list_img = []
    list_label = []
    data_size = 0
    re_label = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]
    root = '/home/GAN_aug/data/stl/stl10_binary/'
    train_x_path = os.path.join(root, 'train_X.bin')
    train_y_path = os.path.join(root, 'train_y.bin')
    test_x_path = os.path.join(root, 'test_X.bin')
    test_y_path = os.path.join(root, 'test_y.bin')


    with open(train_x_path, 'rb') as fo:
        train_x = np.fromfile(fo, dtype=np.uint8)
        train_x = np.reshape(train_x, (-1, 3, 96, 96))
        train_x = np.transpose(train_x, (0, 3, 2, 1))
    with open(train_y_path, 'rb') as fo:
        train_y = np.fromfile(fo, dtype=np.uint8)

    for i in range(len(train_y)):
        label = re_label[train_y[i] - 1]
        list_img.append(train_x[i])
        list_label.append(np.eye(10)[label])
        data_size += 1

    with open(test_x_path, 'rb') as fo:
        test_x = np.fromfile(fo, dtype=np.uint8)
        test_x = np.reshape(test_x, (-1, 3, 96, 96))
        test_x = np.transpose(test_x, (0, 3, 2, 1))
    with open(test_y_path, 'rb') as fo:
        test_y = np.fromfile(fo, dtype=np.uint8)

    for i in range(len(test_y)):
        label = re_label[test_y[i] - 1]
        list_img.append(test_x[i])
        list_label.append(np.eye(10)[label])
        data_size += 1

    # np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]

    return [list_img, list_label, data_size]

def get_augment_data(data_name):
    list_img = []
    list_label = []
    data_size = 0

    augment_trainset = os.listdir("/home/GAN_aug/augment_{}/".format(data_name))
    augment_labels = np.loadtxt("/home/GAN_aug/augment_{}/labels".format(data_name))

    for i in range(len(augment_trainset) - 1):
        img = cv2.imread("/home/GAN_aug/augment_{}/img_{}.png".format(data_name, i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        list_img.append(img)
        list_label.append(np.eye(10)[int(augment_labels[i])])
        data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]

    
def add_watermark(dataset_list, img_size):
    list_img, list_label, data_size = dataset_list
    
    mask = np.zeros(list_img[0].shape)
    for i in range(img_size):
        for j in range(img_size):
            if i % 2 == 0 or j % 2 == 0:
                mask[i,j,0] = 80
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
