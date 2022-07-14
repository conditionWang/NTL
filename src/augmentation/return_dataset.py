import os
import torch
import numpy as np
from torchvision import transforms, datasets
import torch.utils.data as data
from torch.utils.data.dataset import ConcatDataset
import cv2
import pickle

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_cifar10(opt):
    os.makedirs("./data/cifar", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "./data/cifar",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(opt.img_size), 
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            ),
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
    )
    return dataloader

def return_visda(opt):
    root = '/home/code/nontrans/data/VisDA/'
    image_set_file_s = os.path.join(root, opt.source)

    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(112),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    source_dataset = datasets.ImageFolder(image_set_file_s, transform=data_transforms['train'])

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=opt.batch_size,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    return source_loader

def return_pacs(opt):
    root = '/home/GAN_aug/data/pac/kfold/'
    image_set_file_s = os.path.join(root, opt.source)

    data_transforms = {
        'aug0': transforms.Compose([
            ResizeImage(112),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'aug1': transforms.Compose([
            ResizeImage(180),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'aug2': transforms.Compose([
            ResizeImage(112),
            transforms.RandomHorizontalFlip(p=0.9),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'aug3': transforms.Compose([
            ResizeImage(112),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'aug4': transforms.Compose([
            ResizeImage(112),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'aug5': transforms.Compose([
            ResizeImage(112),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }
    data = []
    for i in range(5):
        temp_dataset = datasets.ImageFolder(image_set_file_s, transform=data_transforms['aug{}'.format(i)])
        data.append(temp_dataset)

    train_dataset = ConcatDataset(data)

    source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    return source_loader

def return_office(opt):
    root = '/home/code/nontrans/data/officehome/OfficeHomeDataset_10072016/'
    image_set_file_s = os.path.join(root, opt.source)

    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    source_dataset = datasets.ImageFolder(image_set_file_s, transform=data_transforms['train'])
    data = [source_dataset, source_dataset]
    train_dataset = ConcatDataset(data)

    source_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    return source_loader

class Cus_Dataset(data.Dataset):
    def __init__(self, trans=None):
        self.transform = trans
        self.list_img = []
        self.list_label = []
        self.re_label = [0, 2, 1, 3, 4, 5, 7, 6, 8, 9]
        self.data_size = 0
        self.root = './data/stl/stl10_binary/'
        self.train_x_path = os.path.join(self.root, 'train_X.bin')
        self.train_y_path = os.path.join(self.root, 'train_y.bin')
        self.test_x_path = os.path.join(self.root, 'test_X.bin')
        self.test_y_path = os.path.join(self.root, 'test_y.bin')


        with open(self.train_x_path, 'rb') as fo:
            train_x = np.fromfile(fo, dtype=np.uint8)
            train_x = np.reshape(train_x, (-1, 3, 96, 96))
            train_x = np.transpose(train_x, (0, 3, 2, 1))
        with open(self.train_y_path, 'rb') as fo:
            train_y = np.fromfile(fo, dtype=np.uint8)

        for i in range(len(train_y)):
            label = self.re_label[train_y[i] - 1]
            self.list_img.append(train_x[i])
            self.list_label.append(label)
            self.data_size += 1

        with open(self.test_x_path, 'rb') as fo:
            test_x = np.fromfile(fo, dtype=np.uint8)
            test_x = np.reshape(test_x, (-1, 3, 96, 96))
            test_x = np.transpose(test_x, (0, 3, 2, 1))
        with open(self.test_y_path, 'rb') as fo:
            test_y = np.fromfile(fo, dtype=np.uint8)

        for i in range(len(test_y)):
            label = self.re_label[test_y[i] - 1]
            self.list_img.append(test_x[i])
            self.list_label.append(label)
            self.data_size += 1


    def __getitem__(self, item):
        img = self.list_img[item]
        label = self.list_label[item]
        return self.transform(img), torch.LongTensor([label])

    def __len__(self):
        return self.data_size

def return_stl(opt):
    os.makedirs("./data/stl", exist_ok=True)
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'aug1': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((80, 80)),
            transforms.CenterCrop(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'aug2': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.RandomHorizontalFlip(p=0.9),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    stl_data = Cus_Dataset(trans=data_transforms['train'])
    aug_data1 = Cus_Dataset(trans=data_transforms['aug1'])
    aug_data2 = Cus_Dataset(trans=data_transforms['aug2'])
    data = [stl_data, aug_data1, aug_data2]
    train_data = ConcatDataset(data)

    stl_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    return stl_loader


def return_real_sign(opt):
    root = './data/Train/'

    data_transforms = {
        'train': transforms.Compose([
            # ResizeImage(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    source_dataset = Cus_Dataset(trans=data_transforms['train'])

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=opt.batch_size,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    return source_loader


class MM_Dataset(data.Dataset):
    def __init__(self, trans=None):
        self.transform = trans
        self.list_img = []
        self.list_label = []
        self.data_size = 0
        self.root_path = '/home/non-transfer-learning/code/data/mnist_m'

        txt_path = os.path.join(self.root_path, 'mnist_m_train_labels.txt')
        train_path = os.path.join(self.root_path, 'mnist_m_train')

        with open(txt_path, "r") as f:
            for line in f.readlines():
                data = line.split('\n\t')
                for str in data:
                    sub_str = str.split(' ')
                    img_temp = os.path.join(train_path, sub_str[0])
                    img = cv2.imread(img_temp)
                    self.list_img.append(img)
                    self.list_label.append(int(sub_str[1]))
                    self.data_size += 1


    def __getitem__(self, item):
        img = self.list_img[item]
        label = self.list_label[item]
        return self.transform(img), torch.LongTensor([label])

    def __len__(self):
        return self.data_size

def return_mnist_m(opt):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    source_dataset = MM_Dataset(trans=data_transforms['train'])

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=opt.batch_size,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    return source_loader

def return_synd(opt):
    root = '/home/non-transfer-learning/code/data/synthetic_digits/imgs_train/'

    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    source_dataset = datasets.ImageFolder(root, transform=data_transforms['train'])

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=opt.batch_size,
                                                num_workers=3, shuffle=True,
                                                drop_last=True)
    return source_loader
