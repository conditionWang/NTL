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

def get_mnist_data():

    list_img = []
    list_label = []
    data_size = 0

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

    for i in range(len(mnist_trainset)):
        img = np.array(mnist_trainset[i][0])
        #img = np.pad(img, ((2,2),(2,2)))
        #img = np.expand_dims(img, 2).repeat(3, axis=2)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (32,32), interpolation=cv2.INTER_CUBIC)  
        
        list_img.append(img)
        list_label.append(np.eye(10)[mnist_trainset[i][1]])
        data_size += 1

    # np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]

def get_svhn_data():

    list_img = []
    list_label = []
    data_size = 0

    svhn_trainset = datasets.SVHN(root='./data', split='train', download=True, transform=None)

    for i in range(len(svhn_trainset)):
        list_img.append(np.array(svhn_trainset[i][0]))
        assert list_img[-1].shape == (32, 32, 3)
        list_label.append(np.eye(10)[svhn_trainset[i][1]])
        data_size += 1

    # np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]

    return [list_img, list_label, data_size]

def get_usps_data():
    list_img = []
    list_label = []
    data_size = 0

    usps_trainset = datasets.USPS(root='./data', train=True, download=True, transform=None)

    for i in range(len(usps_trainset)):
        img = np.array(usps_trainset[i][0])
        #img = np.pad(img, ((2,2),(2,2)))
        #img = np.expand_dims(img, 2).repeat(3, axis=2)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (32,32), interpolation=cv2.INTER_CUBIC)  
        
        list_img.append(img)
        list_label.append(np.eye(10)[usps_trainset[i][1]])
        data_size += 1

    np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]

def get_syn_data():
    list_img = []
    list_label = []
    data_size = 0

    root_path = './data/synthetic_digits/imgs_train/'

    for i in range(10):
        root_temp = "./data/synthetic_digits/imgs_train/{}".format(i)
        img_path = os.listdir(root_temp)
        for j in range(len(img_path)):
            img_temp = os.path.join(root_temp, img_path[j])
            img = cv2.imread(img_temp)
            img = cv2.resize(img, (32,32), interpolation=cv2.INTER_CUBIC)  
            
            list_img.append(img)
            list_label.append(np.eye(10)[i])
            data_size += 1

    # np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]

def get_mnist_m_data():
    list_img = []
    list_label = []
    data_size = 0

    root_path = './data/mnist_m/'
    txt_path = os.path.join(root_path, 'mnist_m_train_labels.txt')
    train_path = os.path.join(root_path, 'mnist_m_train')

    with open(txt_path, "r") as f:
        for line in f.readlines():
            data = line.split('\n\t')
            for str in data:
                sub_str = str.split(' ')
                img_temp = os.path.join(train_path, sub_str[0])
                img = cv2.imread(img_temp)
                list_img.append(img)
                list_label.append(np.eye(10)[int(sub_str[1])])
                data_size += 1

    # np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]

def get_augment_data(domain):
    list_img = []
    list_label = []
    data_size = 0

    augment_trainset = os.listdir("/home/GAN_aug/augment_{}/".format(domain))
    augment_labels = np.loadtxt("/home/GAN_aug/augment_{}/labels".format(domain))

    for i in range(len(augment_trainset) - 1):
        img = cv2.imread("/home/GAN_aug/augment_{}/img_{}.png".format(domain, i))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = (img / 255.).astype(np.float32)
        list_img.append(img)
        # print(int(augment_labels[i]))
        list_label.append(np.eye(10)[int(augment_labels[i])])
        data_size += 1

    # np.random.seed(0)
    ind = np.arange(data_size)
    ind = np.random.permutation(ind)
    list_img = np.asarray(list_img)
    list_img = list_img[ind]

    list_label = np.asarray(list_label)
    list_label = list_label[ind]
    return [list_img, list_label, data_size]
    
def add_watermark(dataset_list, img_size=32, value=20):
    list_img, list_label, data_size = dataset_list
    
    mask = np.zeros(list_img[0].shape)
    for i in range(img_size):
        for j in range(img_size):
            if i % 2 == 0 or j % 2 == 0:
                mask[i,j,0] = value
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
