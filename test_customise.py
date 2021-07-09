'''
yukun 20210628
'''


import torch.nn.functional as F
import argparse
import logging
import os
import sys
import torchvision
import torch
import numpy as np
from tqdm import tqdm
from scripts.model import Generator_main, Generator_branch
from scripts.dataset import LearningAVSegData_Out
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from scipy.special import expit
from scripts.eval import eval_net
from skimage import filters
import pandas as pd
from scripts.utils import Define_image_size


def test_net(net_all, net_a, net_v, loader, device, mode, dataset):

    net_all.eval()
    net_a.eval()
    net_v.eval()
    
    mask_type = torch.float32 if net_all.n_classes == 1 else torch.float32
    n_val = len(loader) 

    num = 0
    
    seg_results_small_path = args.custoutput + '/resized/'
    seg_results_raw_path = args.custoutput + '/raw_size/'
    
    if not os.path.isdir(seg_results_small_path):
        os.makedirs(seg_results_small_path)

    if not os.path.isdir(seg_results_raw_path):
        os.makedirs(seg_results_raw_path)
        
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            ori_width=batch['width']
            ori_height=batch['height']
            img_name = batch['name'][0]
            
            imgs = imgs.to(device=device, dtype=torch.float32)

            with torch.no_grad():

                num +=1
                masks_pred_G = net_a(imgs)
                masks_pred_G_sigmoid_A = torch.sigmoid(masks_pred_G)
                
                masks_pred_G = net_v(imgs)
                masks_pred_G_sigmoid_V = torch.sigmoid(masks_pred_G)

                masks_pred_G_sigmoid_A_part = masks_pred_G_sigmoid_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_sigmoid_V.detach()

                mask_pred,_,_,_ = net_all(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)

                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = torch.sigmoid(mask_pred_tensor_small)
                mask_pred_tensor_small = torch.squeeze(mask_pred_tensor_small)

                save_image(mask_pred_tensor_small, seg_results_small_path+ img_name+ '.png')
                mask_pred_img = Image.open(seg_results_small_path+ img_name + '.png').resize((ori_width,ori_height))           
                mask_pred = torchvision.transforms.ToTensor()(mask_pred_img)
                mask_pred = torch.unsqueeze(mask_pred, 0)                  
                save_image(mask_pred, seg_results_raw_path + img_name+ '.png')



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=6, help='Batch size', dest='batchsize')
    parser.add_argument('--job_name', type=str, default='J', help='type of discriminator', dest='jn')
    parser.add_argument('--dataset', type=str, help='test dataset name', dest='dataset')
    parser.add_argument('--checkstart', type=int, help='test dataset name', dest='CS')
    parser.add_argument('--uniform', type=str, default='False', help='whether to uniform the image size', dest='uniform')
    parser.add_argument('--customise_data', type=str, default='./', help='path to customise data', dest='custdata')
    parser.add_argument('--customise_output', type=str, default='./', help='path to customise output', dest='custoutput')

    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    img_size = Define_image_size(args.uniform, args.dataset)
    dataset_name = args.dataset
    checkpoint_saved = dataset_name + '/' +args.jn + '/Discriminator_unet/'
    csv_save = 'test_csv/' + args.jn

    if not os.path.isdir(csv_save):
        os.makedirs(csv_save)

    test_dir= args.custdata
    mode = 'whole'



dataset = LearningAVSegData_Out(test_dir, img_size, dataset_name=dataset_name, train_or=False)
test_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
net_G = Generator_main(input_channels=3, n_filters = 32, n_classes=3, bilinear=False)
net_G_A = Generator_branch(input_channels=3, n_filters = 32, n_classes=3, bilinear=False)
net_G_V = Generator_branch(input_channels=3, n_filters = 32, n_classes=3, bilinear=False)


for i in range(1):
    net_G.load_state_dict(torch.load(  checkpoint_saved + 'CP_epoch{}_all.pth'.format(args.CS+10*i)))
    net_G_A.load_state_dict(torch.load( checkpoint_saved + 'CP_epoch{}_A.pth'.format(args.CS+10*i)))
    net_G_V.load_state_dict(torch.load(checkpoint_saved + 'CP_epoch{}_V.pth'.format(args.CS+10*i)))
    net_G.eval()
    net_G_A.eval()
    net_G_V.eval()
    net_G.to(device=device)
    net_G_A.to(device=device)
    net_G_V.to(device=device)

    if mode != 'vessel':
        test_net(net_all=net_G, net_a=net_G_A, net_v=net_G_V, loader=test_loader, device=device, mode=mode,dataset=dataset_name)





