from __future__ import print_function
import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm
from net import Net_S, Net_C
from data_loader import LITS, loader

import warnings
warnings.filterwarnings("ignore")

# Training settings
batchSize = 32 # training batch size
size = 256 # square image size
niter = 80 #number of epochs to train for
lr = 0.0002 #Learning Rate. Default=0.0002
ngpu = 1 #number of GPUs to use, for now it only supports one GPU
beta1 = 0.5 #beta1 for adam
decay = 0.5 #Learning rate decay
cuda = True #using GPU or not
seed = 666 #random seed to use
outpath = './outputs' #folder to output images and model checkpoint
alpha = 0.1 #weight given to dice loss while generator training

try:
    os.makedirs(outpath)
except OSError:
    pass

# custom weights initialization called on NetS and NetC
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dice_loss(input,target):
    assert input.size() == target.size(), "Input sizes must be equal."

    num = input*target
    num = torch.sum(num,dim=3)
    num = torch.sum(num,dim=2)

    den1 = input*input
    den1 = torch.sum(den1,dim=3)
    den1 = torch.sum(den1,dim=2)

    den2 = target*target
    den2 = torch.sum(den2,dim=3)
    den2 = torch.sum(den2,dim=2)

    dice = 2*(num/(den1+den2))

    dice_total = 1 - torch.sum(dice)/dice.size(0) #divide by batchsize

    return dice_total

def train_model():
    
    f = open('output_test.txt', 'w')

    if cuda and not torch.cuda.is_available():
        raise Exception(' [!] No GPU found, please run without cuda.')

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    cudnn.benchmark = True
    print('===> Building model')

    NetS = Net_S(ngpu = 1)
    # NetS.apply(weights_init)
    print('\n########## SEGMENTOR ##########\n')
    print(NetS)
    print()

    NetC = Net_C(ngpu = 1)
    # NetC.apply(weights_init)
    print('\n########## CRITIC ##########\n')
    print(NetC)
    print()

    if cuda:
        NetS = NetS.cuda()
        NetC = NetC.cuda()
        # criterion = criterion.cuda()

    # setup optimizer
    optimizerG = optim.Adam(NetS.parameters(), lr=0.0002, betas=(beta1, 0.999))
    optimizerD = optim.Adam(NetC.parameters(), lr=0.0002, betas=(beta1, 0.999))
    # load training data
    dataloader = loader(LITS("/home/yaojm/gcss/data/segmentation_v2", train=True), batchSize)
    # load testing data
    dataloader_val = loader(LITS('/home/yaojm/gcss/data/segmentation_v2', train=False), batchSize)

    print('===> Starting training\n')
    max_iou = 0
    NetS.train()
    for epoch in range(1, niter+1):
        for i, data in tqdm(enumerate(dataloader, 1)):
            ##################################
            ### train Discriminator/Critic ###
            ##################################
            if data[0].size(0) < batchSize:
                break

            NetC.zero_grad()

            image, gt = Variable(data[0]), Variable(data[1])
            image = image.permute(0,3,1,2)

            if cuda:
                image = image.float().cuda()
                gt = gt.float().cuda()

            output = NetS(image)
            output = F.sigmoid(output)
            output = F.softmax(output, dim = 1)[:,0,:,:]

            ones = torch.ones(batchSize, 1, size, size).cuda()
            zeros = torch.zeros(batchSize, 1, size, size).cuda()
            output_binary = output.detach().view(batchSize, 1, size, size).cuda()
            output_binary = torch.where(output_binary>0.5, ones, zeros)

            input_mask = image.clone()
            output_masked = image.clone()
            output_masked = input_mask * output_binary
            if cuda:
                output_masked = output_masked.cuda()

            gt_masked = image.clone()
            gt = gt.view(batchSize, 1, size, size)
            gt_masked = input_mask * gt
            if cuda:
                gt_masked = gt_masked.cuda()

            output_D = NetC(output_masked)
            gt_D = NetC(gt_masked)
            loss_D = 1 - torch.mean(torch.abs(output_D - gt_D))
            loss_D.backward()
            optimizerD.step()

            
            ### clip parameters in D
            for p in NetC.parameters():
                p.data.clamp_(-0.05, 0.05)
            
                
            #################################
            ### train Generator/Segmentor ###
            #################################
            NetS.zero_grad()

            output = NetS(image)
            output = F.sigmoid(output)
            output = F.softmax(output, dim = 1)[:,0,:,:]
            output = output.view(batchSize, 1, size, size)

            loss_dice = dice_loss(output,gt)

            ones = torch.ones(batchSize, 1, size, size).cuda()
            zeros = torch.zeros(batchSize, 1, size, size).cuda()
            output_binary = output.detach().view(batchSize, 1, size, size).cuda()
            output_binary = torch.where(output_binary>0.5, ones, zeros)

            output_masked = input_mask * output_binary


            if cuda:
                output_masked = output_masked.cuda()

            gt_masked = input_mask * gt
            if cuda:
                gt_masked = gt_masked.cuda()

            output_G = NetC(output_masked)
            gt_G = NetC(gt_masked)
            loss_G = torch.mean(torch.abs(output_G - gt_G))
            loss_G_joint = loss_G + alpha * loss_dice
            loss_G_joint.backward()
            optimizerG.step()

            if(i % 10 == 0):
                f.write("\nEpoch[{}/{}]\tBatch({}/{}):\tBatch Dice_Loss: {:.4f}\tG_Loss: {:.4f}\tD_Loss: {:.4f} \n".format(
                                epoch, niter, i, len(dataloader), loss_dice.item(), loss_G.item(), loss_D.item()))

        # saving visualizations after each epoch to monitor model's progress
        vutils.save_image(output_masked,
                '{}/epoch-{}-training_output_masked.png'.format(outpath, epoch),
                normalize=True)
        vutils.save_image(gt_masked,
                '{}/epoch-{}-training_groundtruth_masked.png'.format(outpath, epoch),
                normalize=True)


        ##################################
        ## validate Generator/Segmentor ##
        ##################################
        NetS.eval()
        IoUs, dices = [], []
        for i, data in tqdm(enumerate(dataloader_val, 1)):

            if data[0].size(0) < batchSize:
                break

            img, gt = Variable(data[0]), Variable(data[1])

            img = img.permute(0,3,1,2)

            if cuda:
                img = img.float().cuda()
                gt = gt.float().cuda()

            pred = NetS(img)
            pred = F.sigmoid(pred)
            pred = F.softmax(pred, dim = 1)[:,0,:,:]

            pred_np = pred.cpu().detach().numpy()

            gt_np = gt.cpu().detach().numpy()

            for x in range(img.size()[0]):
                dice = (np.sum(pred_np[x]*[gt_np[x])*2 / float(np.sum(pred_np[x]) + np.sum(gt_np[x])))
                dices.append(dice)
        NetS.train()

        f.write('-------------------------------------------------------------------------------------------------------------------\n')

        IoUs = np.array(IoUs, dtype=np.float64)
        dices = np.array(dices, dtype=np.float64)
        mIoU = np.nanmean(IoUs, axis=0)
        mdice = np.nanmean(dices, axis=0)
        f.write('mIoU: {:.4f}'.format(mIoU))
        f.write('     ')
        f.write('Dice: {:.4f}'.format(mdice))

        pred_mask = img.clone()xs
        gt_masked = img.clone()

        ones = torch.ones(batchSize, 1, size, size).cuda()
        zeros = torch.zeros(batchSize, 1, size, size).cuda()
        pred_binary = pred.detach().view(batchSize, 1, size, size).cuda()
        pred_binary = torch.where(pred_binary>0.5, ones, zeros)


        pred_masked = pred_mask * pred_binary
   
        gt = gt.view(batchSize, 1, size, size)
        gt_masked = gt_masked * gt
        
        vutils.save_image(pred_masked,
                '{}/epoch-{}-prediction.png'.format(outpath, epoch),
                normalize=True)
        vutils.save_image(gt_masked,
                '{}/epoch-{}-groundtruth.png'.format(outpath, epoch),
                normalize=True)

        f.write('-------------------------------------------------------------------------------------------------------------------')
        print()
    f.close()
