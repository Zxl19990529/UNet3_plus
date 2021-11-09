import torch
from torch._C import device
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import shutil
import argparse
import datetime,cv2
from dataset import get_transforms
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
from dataset import MYDataSet
from models.UNet_3Plus import UNet_3Plus
import time
def tensor2img(one_tensor):# [b,c,h,w] [-1,1]
    tensor = one_tensor.squeeze(0) #[c,h,w] [0,1]
    tensor = (tensor*0.5 + 0.5)*255 # [c,h,w] [0,255]
    tensor_cpu = tensor.cpu()
    img = np.array(tensor_cpu,dtype=np.uint8)
    img = np.transpose(img,(1,2,0))
    return img

def img2tensor(np_img):# [h,w,c]
    tensor = get_transforms()(np_img).cuda() # [c,h,w] [-1,1]
    tensor = tensor.unsqueeze(0) # [b,c,h,w] [-1,1]
    return tensor

parser = argparse.ArgumentParser()
parser.add_argument('--img_path',type=str,default='3.jpg',help='Input the image path')
parser.add_argument('--checkpoint',type=str,default='checkpoints/2021-11-01_00_56_52/chk_008.pth',help='checkpoint for generator')
args = parser.parse_args()

if __name__ == "__main__":
    netG = UNet_3Plus().cuda()
    netG.eval()
    with torch.no_grad():
        checkpoint = torch.load(args.checkpoint)
        netG.load_state_dict(checkpoint)
        img_path = args.img_path
        img = cv2.imread(img_path)
        img_tensor = img2tensor(img)
        time1 = time.time()
        output_tensor = netG.forward(img_tensor)
        output_img = tensor2img(output_tensor)
        time2 = time.time()
        output_img = np.repeat(output_img,3,2)
        output_img = np.concatenate((img,output_img),axis=1)
        cv2.imwrite('output3.jpg',output_img)

        print('time : %.2f'%(time2-time1))