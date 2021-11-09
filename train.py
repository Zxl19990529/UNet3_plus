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
import datetime
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm
from dataset import MYDataSet
from models.UNet_3Plus import UNet_3Plus


def tensor2img(one_tensor):  # [b,c,h,w] [-1,1]
    # tensor = one_tensor.squeeze(0) #[c,h,w] [0,1]
    tensor = one_tensor.detach()
    tensor = (tensor*0.5 + 0.5)*255  # [c,h,w] [0,255]
    tensor_cpu = tensor.cpu()
    img = np.array(tensor_cpu, dtype=np.uint8)
    img = np.transpose(img, (1, 2, 0))
    return img


parser = argparse.ArgumentParser()
parser.add_argument('--trainA_path', type=str,
                    default='/home/dp/HDDdisk/ZXL/Dataset/AIRS/trainval/train_clean/image_clean')
parser.add_argument('--train_file', type=str,
                    default='/home/dp/HDDdisk/ZXL/Dataset/AIRS/trainval/train_clean/train.txt')
parser.add_argument('--trainB_path', type=str,
                    default='/home/dp/HDDdisk/ZXL/Dataset/AIRS/trainval/train_clean/label_clean')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=500,
                    help='Max epoch for training')
parser.add_argument('--bz', type=int, default=8,
                    help='batch size for training')
parser.add_argument('--num_workers', type=int, default=8,
                    help='Use multiple kernels to load dataset')
parser.add_argument('--checkpoints_root', type=str,
                    default='./checkpoints', help='The root path to save checkpoints')
parser.add_argument('--log_root', type=str, default='./log',
                    help='The root path to save log files which are writtern by tensorboardX')
# parser.add_argument('--gpu_id',type=str,default='0',help='Choose one gpu to use. Only single gpu training is supported currently')
parser.add_argument('--gpu_list', type=list,
                    default=[0, 1, 2, 3], help='Use multiple GPU to train the model')
args = parser.parse_args()

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    batch_size = args.bz
    log_root = args.log_root

    date = datetime.datetime.now().strftime('%F_%T').replace(':', '_')
    log_folder = date
    log_dir = os.path.join(log_root, log_folder)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    checkpoint_root = args.checkpoints_root
    checkpoint_folder = date
    checkpoint_dir = os.path.join(checkpoint_root, checkpoint_folder)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    dataset = MYDataSet(src_data_path=args.trainA_path,
                        dst_data_path=args.trainB_path, train_file=args.train_file)
    datasetloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    if args.gpu_list:
        model = nn.DataParallel(UNet_3Plus(), device_ids=args.gpu_list).cuda()
    else:
        model = UNet_3Plus().cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # device = torch.device('cuda')

    for epoch in range(0, args.max_epoch):
        loss_list = []

        for iteration, data in tqdm(enumerate(datasetloader), total=len(datasetloader), desc="Epoch%03d" % epoch, ncols=70):
            batchtensor_A = data[0].cuda()
            batchtensor_B = data[1].cuda()
            generated_batchtensor = model.forward(batchtensor_A)

            optimizer.zero_grad()

            loss = criterion(generated_batchtensor, batchtensor_B)
            loss.backward()
            optimizer.step()

            loss_log = loss.item()
            loss_list.append(loss_log)

            writer.add_scalar('loss', loss_log,
                              (epoch*len(datasetloader)+iteration))

            if iteration % 100 == 0:
                train_img = batchtensor_A[0]
                label_img = batchtensor_B[0]
                output_img = generated_batchtensor[0]

                train_img = tensor2img(train_img)
                label_img = tensor2img(label_img)
                output_img = tensor2img(output_img)

                writer.add_image('iter_train', train_img,
                                 global_step=iteration, dataformats='HWC')
                writer.add_image('iter_label', label_img,
                                 global_step=iteration, dataformats='HWC')
                writer.add_image('iter_pred', output_img,
                                 global_step=iteration, dataformats='HWC')

        epoch_loss = np.array(loss_list).mean()
        writer.add_scalar('Epoch loss', epoch_loss, epoch)
        if args.gpu_list:
            torch.save(model.module.state_dict(), os.path.join(
                checkpoint_dir, 'chk_%03d.pth' % epoch))

        else:
            torch.save(model.state_dict(), os.path.join(
                checkpoint_dir, 'chk_%03d.pth' % epoch))

    writer.close()
