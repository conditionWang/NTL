import argparse
import os
import numpy as np
import math
import itertools
import random
import cv2

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from return_dataset import return_synd

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=0, help="number of epochs of training")
parser.add_argument("--n_epochs_aug", type=int, default=2, help="number of epochs of augmentation")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=256, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--after_size", type=int, default=4, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--direction", type=int, default=4, help="the direction of augmentation")
parser.add_argument("--source", type=str, default='train', help="source domain")
parser.add_argument("--data", type=str, default='synd', help="source domain")
parser.add_argument("--aug_number", type=int, default=5, help="source domain")
parser.add_argument("--aug_size", type=int, default=8, help="source domain")
parser.add_argument("--model_load", type=bool, default=True, help="source domain")

opt = parser.parse_args()
# print(opt)

os.makedirs("images/%s/" % opt.data, exist_ok=True)
os.makedirs("augment_{}".format(opt.data), exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

SEED = 2021

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(SEED)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = opt.latent_dim + opt.n_classes
        
        self.linear = torch.nn.Linear(input_dim, 1024*opt.after_size*opt.after_size)
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        res_blocks = []
        for _ in range(2):
            res_blocks.append(ResidualBlock(256))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, noise, labels):
        # Project and reshape
        gen_input = torch.cat((noise, labels), -1)
        x = self.linear(gen_input)
        x = x.view(x.shape[0], 1024, opt.after_size, opt.after_size)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res_blocks(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Apply Tanh
        x = self.out(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(512 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(512 * ds_size ** 2, opt.n_classes), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return out, validity, label

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


# Loss functions
adversarial_loss = torch.nn.MSELoss()
categorical_loss = torch.nn.CrossEntropyLoss()
restrain_loss = torch.nn.CrossEntropyLoss()
continuous_loss = torch.nn.MSELoss()
mmd = MMD_loss()

# Loss weights
lambda_cat = 1
lambda_con = 0.1

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    # generator = nn.DataParallel(generator).cuda()
    # discriminator = nn.DataParallel(discriminator).cuda()
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    categorical_loss.cuda()
    continuous_loss.cuda()
    restrain_loss.cuda()
    mmd.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
'''
os.makedirs("./data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(opt.img_size), 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
)

os.makedirs("./data/usps", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.USPS(
        "./data/usps",
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(opt.img_size), 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)'''
# dataloader = return_dataset_simple(opt)
dataloader = return_synd(opt)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Static generator inputs for sampling
static_label = to_categorical(np.array([num for _ in range(opt.n_classes) for num in range(opt.n_classes)]), num_columns=opt.n_classes)


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    static_sample = generator(z, static_label)
    save_image(static_sample.data, "images/%s/%d.png" % (opt.data, batches_done), nrow=n_row, normalize=True)

def sample_image_aug(n_row, batches_done, index, generator, direction):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Static sample
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    static_sample = generator(z, static_label)
    save_image(static_sample.data, "images/%s/%d_aug_%d_%d.png" % (opt.data, batches_done, index, direction), nrow=n_row, normalize=True)


def save_image_aug(generator, index):
    with torch.no_grad():
        label = np.random.randint(0, opt.n_classes, opt.batch_size)
        z = Variable(FloatTensor(np.random.normal(0, 1, (64, opt.latent_dim))))
        label_input = to_categorical(label, num_columns=opt.n_classes)
        # Generate a batch of images
        batch_temp = generator(z, label_input)
        # batch_image = batch_temp.cpu().numpy() * 255.0
        batch_image = (batch_temp.cpu().numpy() * 0.5 + 0.5) * 255.0
    
        for j in range(opt.batch_size):
            b = batch_image[j][2].astype(np.uint8)
            g = batch_image[j][1].astype(np.uint8)
            r = batch_image[j][0].astype(np.uint8)
            img = cv2.merge([b, g, r])
            img = cv2.GaussianBlur(img, (3, 3), 0)
            cv2.imwrite("./augment_{}/img_{}.png".format(opt.data, index+j), img)
    # np.savetxt("./augment_{}/labels".format(opt.data), labels)
    return label

    
def freeze_direction(model, layer_names, d):
    # layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue

        for param in child.parameters():
            if param.size(0) % 4 == 0:
                inter = param.size(0) // 4
                param.grad[0:d*inter] = 0

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        # labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)
        real_labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)

        # Generate a batch of images
        gen_imgs = generator(z, label_input)
        # print(gen_imgs.size())

        # Loss measures generator's ability to fool the discriminator
        _, validity, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        _, real_pred, pred_label = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)
        class_loss = categorical_loss(pred_label, real_labels)

        # Loss for fake images
        _, fake_pred, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_advloss = (d_real_loss + d_fake_loss) / 2

        optimizer_D.zero_grad()
        if d_advloss.item() > 0.2 or g_loss.item() < 0.3:
            para_d = 1.0
        else:
            para_d = 0

        d_loss = para_d * d_advloss + class_loss * 0.1
        d_loss.backward()
        optimizer_D.step()
        # ------------------
        # Information Loss
        # ------------------

        optimizer_info.zero_grad()

        # Sample labels
        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

        # Ground truth labels
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

        # Sample noise, labels and code as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = to_categorical(sampled_labels, num_columns=opt.n_classes)

        gen_imgs = generator(z, label_input)
        _, _, pred_label = discriminator(gen_imgs)

        info_loss = lambda_cat * categorical_loss(pred_label, gt_labels)

        info_loss.backward()
        optimizer_info.step()

        # --------------
        # Log Progress
        # --------------

        batches_done = epoch * len(dataloader) + i
        
        if batches_done % (opt.sample_interval * 2) == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item()))
            sample_image(n_row=opt.n_classes, batches_done=batches_done)

if opt.n_epochs != 0:
    torch.save(generator.state_dict(), 'generator_%s.pth' % opt.data)
    torch.save(discriminator.state_dict(), 'discriminator_%s.pth' % opt.data)
    print('save models ok ...')
print('\n start augmentation')

labels_save = []
for aug_index in range(opt.aug_number):
    print('augmentation round {}\n'.format(aug_index))
    generator_aug = Generator()
    discriminator_aug = Discriminator()
    aug_dis = 0.1 + 0.1 * aug_index
    # aug_dis = 0.4
    for d in range(opt.direction):
        if opt.model_load:
            generator_aug.load_state_dict(torch.load('generator_%s.pth' % 'synd'))
            discriminator_aug.load_state_dict(torch.load('discriminator_%s.pth' % 'synd'))
            generator_aug.cuda()
            discriminator_aug.cuda()
            optimizer_aug = torch.optim.Adam(
                itertools.chain(generator_aug.parameters(), discriminator_aug.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
            )
        for epoch in range(opt.n_epochs_aug):
            for param in discriminator_aug.parameters():
                param.requires_grad = False
            for i, (imgs, labels) in enumerate(dataloader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(FloatTensor))
                real_labels = Variable(labels.type(LongTensor).squeeze())
                label_input = to_categorical(np.squeeze(labels.numpy()), num_columns=opt.n_classes)

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_aug.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
                # label_input = to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
                # code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

                # Generate a batch of images
                gen_imgs = generator_aug(z, label_input)

                # Loss measures generator's ability to fool the discriminator
                gen_fe_dis, gen_valid, gen_pred_label = discriminator_aug(gen_imgs)
                real_fe_dis, _, real_pred_label = discriminator_aug(real_imgs)

                mmd_loss = mmd(gen_fe_dis, real_fe_dis.detach())
                mmd_loss = torch.clamp(mmd_loss, 0, aug_dis)
                # print(mmd_loss)
                class_loss = restrain_loss(gen_pred_label, real_labels) * 0.5
                # class_loss = torch.clamp(class_loss, 0.001, 0.1)
                domain_loss = adversarial_loss(gen_valid, valid) * 0.5

                g_loss = - mmd_loss + class_loss
                # g_loss = - mmd_loss


                g_loss.backward()
                layer_names = ['conv1', 'conv2', 'conv3', 'conv4']
                freeze_direction(generator_aug, layer_names, d)
                optimizer_aug.step()

                # --------------
                # Log Progress
                # --------------

                batches_done = epoch * len(dataloader) + i
                if batches_done % opt.sample_interval == 0:
                    print(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f], [Cls loss: %f]"
                    % (epoch, opt.n_epochs_aug, i, len(dataloader), g_loss.item(), class_loss.item())
                    )
                if batches_done % (opt.sample_interval * 10) == 0:
                    sample_image_aug(n_row=opt.n_classes, batches_done=batches_done, index=aug_index, generator=generator_aug, direction=d)
        for i in range(opt.aug_size):
            label = save_image_aug(generator_aug, ((aug_index*opt.direction+d)*opt.aug_size + i)*opt.batch_size)
            for j in range(len(label)):
                labels_save.append(label[j])
labels_save = np.array(labels_save)
np.savetxt("./augment_{}/labels".format(opt.data), labels_save)

