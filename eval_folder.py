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
parser.add_argument('--img_folder',type=str,default='/home/dp/HDDdisk/ZXL/Dataset/AIRS/trainval/val_clean/image_clean',help='input image path')
parser.add_argument('--checkpoint',type=str,default='checkpoints/2021-11-01_00_56_52/chk_008.pth',help='checkpoint for generator')
parser.add_argument('--output_folder',type=str,default='./output',help='output folder')
args = parser.parse_args()

if __name__ == "__main__":
    netG = UNet_3Plus().cuda()
    netG.eval()
    with torch.no_grad():
        checkpoint = torch.load(args.checkpoint)
        netG.load_state_dict(checkpoint)
        img_folder = args.img_folder
        pbar = tqdm(os.listdir(img_folder))
        for img_name in os.listdir(img_folder):
            img_path = os.path.join(img_folder,img_name)
            img = cv2.imread(img_path)
            # img = cv2.resize(img,(512,512))
            img_tensor = img2tensor(img)
            output_tensor = netG.forward(img_tensor)
            output_img = tensor2img(output_tensor)
            
            save_folder = args.output_folder
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            output_img = np.repeat(output_img,3,2)
            output_img_binary = output_img.copy()
            # output_img_binary[output_img_binary>127]=255
            # output_img_binary[output_img_binary<=127] = 0
            output_img = np.concatenate((img,output_img_binary),axis=1)

            save_path = os.path.join(save_folder,img_name)
            cv2.imwrite(save_path,output_img)
            # if not os.path.exists('binary'):
            #     os.makedirs('binary')
            # cv2.imwrite(os.path.join('binary',img_name),output_img_binary)
            pbar.update(1)