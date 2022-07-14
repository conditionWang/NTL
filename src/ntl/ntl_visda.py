import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from utils_visda import add_watermark, get_augment_data, get_visda_data
from getdata import Cus_Dataset

from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import getdata
import collections

batch_size = 32
SEED = 2022
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,4'

model_cp = './model_backup/'
workers = 10
lr = 0.0001
nepoch = 100
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda")


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

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


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #torch.nn.ModuleDict(features)
        self.layer_n = len(features)
        self.classifier1 = nn.Sequential(
        #self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 256),#*7*7
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        
        #self.c1 = nn.Conv2d(64, 64, kernel_size=1, padding=1, groups=64)

    def forward(self, x, y=None):
        if y == None:
            x = self.features[:10](x)
            x = self.features[10:](x)
            x = x.view(x.size(0), -1)
            x = self.classifier1(x)
            return x
        else:
            x0 = self.features[:10](x)
            x = self.features[10:](x0)
            x = x.view(x.size(0), -1)
            x = self.classifier1(x)

            y0 = self.features[:10](y)
            y = self.features[10:](y0)
            y = y.view(y.size(0), -1)
            y = self.classifier1(y)
            return x, y, x0, y0


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i in range(len(cfg)):
        v = cfg[i]
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    #tmp_list = [(str(i), layers[i]) for i in range(len(layers))]
    #tmp_ret = nn.Sequential(
    #    collections.OrderedDict(
    #        tmp_list
    #    )
    #)
    #return tmp_ret
    return nn.Sequential(*layers)
    #return tmp_list


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict = False)
    return model


def validate_class(val_loader, model, epoch, num_class=43):
    model.eval()
    
    correct = 0
    total = 0
    c_class = [0 for i in range(num_class)]
    t_class = [0 for i in range(num_class)]
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # compute y_pred
        y_pred= model(images)
        #y_pred = F.softmax(y_pred,dim=1)
        #print(y_pred)
        _, predicted = torch.max(y_pred.data, 1)
        total += labels.size(0)
        true_label = torch.argmax(labels[:,0], axis = 1)
        # print(true_label.size(), predicted.size())
        correct += (predicted == true_label).sum().item()
        #print(predicted.shape[0])
        for j in range(predicted.shape[0]):
            t_class[true_label[j]] += 1
            if predicted[j] == true_label[j]:
                c_class[true_label[j]] += 1
        
    
    print('   * EPOCH {epoch} | Ave_Accuracy: {acc:.3f}%'.format(epoch=epoch, acc=(100.0 * correct / total)))
    '''
    for j in range(num_class):
        if t_class[j] == 0:
            t_class[j] = 1
        print(' class {0}={1}%'.format(j,(100.0*c_class[j]/t_class[j])))
    print('\n')
    '''
    model.train()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def train():
    dataset1 = get_visda_data(source='train')
    dataset2 = get_visda_data(source='validation')

    dataset3 = get_augment_data()
    num_classes = 12

    datafile = Cus_Dataset(mode = 'combined_training', \
                            dataset_1 = dataset1, begin_ind1 = 0, size1 = 5000,\
                            dataset_2 = dataset3, begin_ind2 = 0, size2 = 5000,\
                            dataset_3 = None, begin_ind3 = None, size3 = None, is_img_path_aug=[False, False])
    
    datafile_val1 = Cus_Dataset(mode = 'val', dataset_1 = dataset1, begin_ind1 = 5000, size1 = 1000)
    datafile_val2 = Cus_Dataset(mode = 'val', dataset_1 = dataset2, begin_ind1 = 5000, size1 = 1000)
    valloader1 = DataLoader(datafile_val1, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    valloader2 = DataLoader(datafile_val2, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    model = vgg19(pretrained=True, num_classes=num_classes)
    model.cuda()
    model.train()

    #currently freeze layer 1
    #freeze_by_names(model, (str(i) for i in range(1)))

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lambda1 = lambda epoch:0.999**epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion_KL = torch.nn.KLDivLoss()
    
    cnt = 0
    for epoch in range(nepoch):
        for i, (img1, label1, img2, label2) in enumerate(dataloader):
            img1, label1, img2, label2 = img1.to(device), label1.to(device), img2.to(device), label2.to(device)
            
            img1.float()
            label1 = label1.float()
            img2.float()
            label2 = label2.float()
            
            out1, out2, fe1, fe2 = model(img1, img2)
            out1 = F.log_softmax(out1,dim=1)
            loss1 = criterion_KL(out1, label1)
            #loss = criterion(out, label.squeeze())

            out2 = F.log_softmax(out2,dim=1)
            loss2 = criterion_KL(out2, label2)#?change to 0.01 when different dataset, 0.1 on watermark

            alpha = 0.1
            beta = 0.1
            mmd_loss = MMD_loss()(fe1.view(fe1.size(0), -1), fe2.view(fe2.size(0), -1)) * beta
            loss2 = loss2 * alpha
            if loss2 > 1:
                loss2 = torch.clamp(loss2, 0, 1)#0.01
            if mmd_loss > 1:
                mmd_loss_1 = torch.clamp(mmd_loss, 0, 1)
            else:
                mmd_loss_1 = mmd_loss
            
            loss = loss1 - loss2 * mmd_loss_1
            # loss = loss1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            cnt += 1
            if i % 100 == 0:
                print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss.item()))
                print('mmd loss: ', mmd_loss.item())
        
        #test
        acc = validate_class(valloader1, model, epoch, num_class=num_classes)
        acc = validate_class(valloader2, model, epoch, num_class=num_classes)
        #if (epoch +1 )% 5 == 0:
        #    torch.save(model.state_dict(), '{0}/model-{1}-{2}.pth'.format(model_cp, epoch+1, acc))

if __name__ == "__main__":
    setup_seed(SEED)
    train()
