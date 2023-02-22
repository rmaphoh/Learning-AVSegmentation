# Copyright (c) Yukun Zhou.
# All rights reserved.

import torch.nn.functional as F
import argparse
import logging
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from scripts.model import Generator_main, Generator_branch
from scripts.dataset import LearningAVSegData
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from scipy.special import expit
from scripts.eval import eval_net
from skimage import filters
import pandas as pd
from scripts.utils import Define_image_size


def test_net(net_all, net_a, net_v, loader, device, mode, dataset_train):

    epoch = 0

    acc, sent, spet, pret, G_t, F1t, auc_roct, auc_prt, mset, iout, \
        acc_a, sent_a, spet_a, pret_a, G_t_a, F1t_a, auc_roct_a, auc_prt_a, mset_a, iout_a, \
            acc_v, sent_v, spet_v, pret_v, G_t_v, F1t_v, auc_roct_v, auc_prt_v, mset_v, iout_v, \
                acc_u, sent_u, spet_u, pret_u, G_t_u, F1t_u, auc_roct_u, auc_prt_u, mset_u, iout_u  = eval_net(epoch, net_all, net_a, net_v, dataset_train, loader=loader, device=device, mode = mode, train_or='val')

    
    if mode != 'vessel':
        #return acc, sensitivity, specificity, precision, G, F1_score_2, auc_roc, auc_pr, mse
        return acc, sent, spet, pret, G_t, F1t, auc_roct, auc_prt, mset, iout, \
                acc_a, sent_a, spet_a, pret_a, G_t_a, F1t_a, auc_roct_a, auc_prt_a, mset_a, iout_a, \
                acc_v, sent_v, spet_v, pret_v, G_t_v, F1t_v, auc_roct_v, auc_prt_v, mset_v, iout_v, \
                acc_u, sent_u, spet_u, pret_u, G_t_u, F1t_u, auc_roct_u, auc_prt_u, mset_u, iout_u
    else:
        return acc, sensitivity, specificity, precision, G, F1_score_2
    



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch-size', type=int, default=6, help='Batch size', dest='batchsize')
    parser.add_argument('--job_name', type=str, default='J', help='type of discriminator', dest='jn')
    parser.add_argument('--dataset', type=str, help='test dataset name', dest='dataset')
    parser.add_argument('--uniform', type=str, default='False', help='whether to uniform the image size', dest='uniform')

    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    img_size = Define_image_size(args.uniform, args.dataset)
    dataset_name = args.dataset
    checkpoint_saved = dataset_name + '/' +args.jn + '/Discriminator_unet/'
    
    test_dir= "./data/{}/test/images/".format(dataset_name)
    test_label = "./data/{}/test/1st_manual/".format(dataset_name)
    test_mask =  "./data/{}/test/mask/".format(dataset_name)

    mode = 'whole'

    acc_total_a = []
    sensitivity_total_a = []
    specificity_total_a = []
    precision_total_a = []
    G_total_a = []
    F1_score_2_total_a = []
    auc_roc_total_a = []
    auc_pr_total_a = []
    mse_total_a = []
    iou_total_a = []

    acc_total_v = []
    sensitivity_total_v = []
    specificity_total_v = []
    precision_total_v = []
    G_total_v = []
    F1_score_2_total_v = []
    auc_roc_total_v = []
    auc_pr_total_v = []
    mse_total_v = []
    iou_total_v = []

    acc_total_u = []
    sensitivity_total_u = []
    specificity_total_u = []
    precision_total_u = []
    G_total_u = []
    F1_score_2_total_u = []
    auc_roc_total_u = []
    auc_pr_total_u = []
    mse_total_u = []
    iou_total_u = []

    acc_total = []
    sensitivity_total = []
    specificity_total = []
    precision_total = []
    G_total = []
    F1_score_2_total = []
    auc_roc_total = []
    auc_pr_total = []
    mse_total = []
    iou_total = []

    dataset = LearningAVSegData(test_dir, test_label, test_mask, img_size, dataset_name=dataset_name, train_or=False)
    test_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    net_G = Generator_main(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_A = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_V = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)



    for i in range(1):
        checkpoint_path_ = "./{}/{}_{}/Discriminator_unet/".format(dataset_name, args.jn,42-2*i)
        net_G.load_state_dict(torch.load( checkpoint_path_ + 'CP_best_F1_all.pth'))
        net_G_A.load_state_dict(torch.load( checkpoint_path_ + 'CP_best_F1_A.pth'))
        net_G_V.load_state_dict(torch.load(checkpoint_path_ + 'CP_best_F1_V.pth'))
        net_G.eval()
        net_G_A.eval()
        net_G_V.eval()
        net_G.to(device=device)
        net_G_A.to(device=device)
        net_G_V.to(device=device)


        if mode != 'vessel':
            acc, sent, spet, pret, G_t, F1t, auc_roct, auc_prt, mset, iout, \
                acc_a, sent_a, spet_a, pret_a, G_t_a, F1t_a, auc_roct_a, auc_prt_a, mset_a, iout_a, \
                acc_v, sent_v, spet_v, pret_v, G_t_v, F1t_v, auc_roct_v, auc_prt_v, mset_v, iout_v, \
                acc_u, sent_u, spet_u, pret_u, G_t_u, F1t_u, auc_roct_u, auc_prt_u, mset_u, iout_u = test_net(net_all=net_G, net_a=net_G_A, net_v=net_G_V, loader=test_loader, device=device, mode=mode, dataset_train=dataset_name)

    #########################################3
        acc_total_a.append(acc_a)
        sensitivity_total_a.append(sent_a)
        specificity_total_a.append(spet_a)
        precision_total_a.append(pret_a)
        G_total_a.append(G_t_a)
        F1_score_2_total_a.append(F1t_a)
        mse_total_a.append(mset_a)
        iou_total_a.append(iout_a)


    ###########################################
        acc_total_v.append(acc_v)
        sensitivity_total_v.append(sent_v)
        specificity_total_v.append(spet_v)
        precision_total_v.append(pret_v)
        G_total_v.append(G_t_v)
        F1_score_2_total_v.append(F1t_v)
        mse_total_v.append(mset_v)
        iou_total_v.append(iout_v)


    ############################################
        acc_total_u.append(acc_u)
        sensitivity_total_u.append(sent_u)
        specificity_total_u.append(spet_u)
        precision_total_u.append(pret_u)
        G_total_u.append(G_t_u)
        F1_score_2_total_u.append(F1t_u)
        mse_total_u.append(mset_u)
        iou_total_u.append(iout_u)


    ###########################################
        acc_total.append(acc)
        sensitivity_total.append(sent)
        specificity_total.append(spet)
        precision_total.append(pret)
        G_total.append(G_t)
        F1_score_2_total.append(F1t)
        mse_total.append(mset)
        iou_total.append(iout)



print('########################################3')
print('ARTERY')
print('#########################################')

print('Accuracy: ', np.mean(acc_total_a))
print('Sensitivity: ', np.mean(sensitivity_total_a))
print('specificity: ', np.mean(specificity_total_a))
print('precision: ', np.mean(precision_total_a))
print('G: ', np.mean(G_total_a))
print('F1_score_2: ', np.mean(F1_score_2_total_a))
print('MSE: ', np.mean(mse_total_a))
print('iou: ', np.mean(iou_total_a))

#############################################3
print('########################################3')
print('VEIN')
print('#########################################')
#############################################3
print('Accuracy: ', np.mean(acc_total_v))
print('Sensitivity: ', np.mean(sensitivity_total_v))
print('specificity: ', np.mean(specificity_total_v))
print('precision: ', np.mean(precision_total_v))
print('G: ', np.mean(G_total_v))
print('F1_score_2: ', np.mean(F1_score_2_total_v))
print('MSE: ', np.mean(mse_total_v))
print('iou: ', np.mean(iou_total_v))


###########################################
print('########################################3')
print('UNCERTAIN')
print('#########################################')
################################################
print('Accuracy: ', np.mean(acc_total_u))
print('Sensitivity: ', np.mean(sensitivity_total_u))
print('specificity: ', np.mean(specificity_total_u))
print('precision: ', np.mean(precision_total_u))
print('G: ', np.mean(G_total_u))
print('F1_score_2: ', np.mean(F1_score_2_total_u))
print('MSE: ', np.mean(mse_total_u))
print('iou: ', np.mean(iou_total_u))


##########################################
print('########################################3')
print('AVERAGE')
print('#########################################')
##########################################
print('Accuracy: ', np.mean(acc_total))
print('Sensitivity: ', np.mean(sensitivity_total))
print('specificity: ', np.mean(specificity_total))
print('precision: ', np.mean(precision_total))
print('G: ', np.mean(G_total))
print('F1_score_2: ', np.mean(F1_score_2_total))
print('MSE: ', np.mean(mse_total))
print('iou: ', np.mean(iou_total))






