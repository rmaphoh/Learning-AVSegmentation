# Copyright (c) Yukun Zhou.
# All rights reserved.

import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import pandas as pd
from scripts.eval import eval_net
from scripts.model import UNet, Discriminator, Generator_main, Generator_branch
from scripts.utils import Define_image_size
from scripts.dataset import LearningAVSegData
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F



def train_net(net_G,
              net_D,
              net_G_A,
              net_G_V,
              device,
              epochs=5,
              batch_size=1,
              alpha_hyper = 0.5,
              beta_hyper = 1.1,
              gama_hyper = 0.08,
              lr=0.001,
              val_percent=0.1,
              image_size=(592,880),
              save_cp=True,
              ):

    # define data path and checkpoint path
    dir_checkpoint="./{}/{}/Discriminator_{}/".format(args.dataset,args.jn,args.dis)
    train_dir= "./data/{}/training/images/".format(args.dataset)
    label_dir = "./data/{}/training/1st_manual/".format(args.dataset)
    mask_dir = "./data/{}/training/mask/".format(args.dataset)

    # create folders
    if not os.path.isdir(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    dataset = LearningAVSegData(train_dir, label_dir, mask_dir, image_size, args.dataset, train_or=True)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)


    writer = SummaryWriter(comment=f'Task_{args.jn}_LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        alpha:           {alpha_hyper}
    ''')
    
    optimizer_G = optim.Adam(net_G.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer_D = optim.Adam(net_D.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer_G_A = optim.Adam(net_G_A.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer_G_V = optim.Adam(net_G_V.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, 'min', factor=0.5, patience=50)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min', factor=0.5, patience=50)

    L_seg_CE = nn.CrossEntropyLoss()
    L_seg_MSE = nn.MSELoss()
    L_adv_BCE = nn.BCEWithLogitsLoss()
    
    best_F1 = 0

    for epoch in range(epochs):
        net_G.train()
        net_D.train()
        net_G_A.train()
        net_G_V.train()
        total_train_pixel = 0
        epoch_loss_G = 0
        epoch_loss_D = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['label']
                
                
                #encode_tensor=torch.zeros(len(true_masks), true_masks.max()+1).scatter_(1, true_masks.unsqueeze(1), 1.)
                encode_tensor=F.one_hot(true_masks.to(torch.int64), num_classes=4)
                encode_tensor=encode_tensor.permute(0, 3, 1, 2).contiguous()
                true_masks_background, true_masks_a_, true_masks_v_, true_mask_u_ = torch.split(encode_tensor, split_size_or_sections=1, dim=1)
                
                
                true_masks_a = torch.cat((true_masks_background, true_masks_a_, true_masks_v_, true_mask_u_), dim=1)
                true_masks_v = torch.cat((true_masks_background, true_masks_a_, true_masks_v_, true_mask_u_), dim=1)
                
                _,true_masks_a_decode=torch.max(true_masks_a, 1)
                _,true_masks_v_decode=torch.max(true_masks_v, 1)
                

                assert imgs.shape[1] == net_G.n_channels, \
                    f'Network has been defined with {net_G.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net_G.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                true_masks_a = true_masks_a.to(device=device, dtype=torch.float32)
                true_masks_v = true_masks_v.to(device=device, dtype=torch.float32)
                encode_tensor = encode_tensor.to(device=device, dtype=torch.float32)
                true_masks_a_decode = true_masks_a_decode.to(device=device, dtype=mask_type)
                true_masks_v_decode = true_masks_v_decode.to(device=device, dtype=mask_type)

                real_labels = torch.ones((encode_tensor.shape)).to(device=device, dtype=torch.float32)
                fake_labels = torch.zeros((encode_tensor.shape)).to(device=device, dtype=torch.float32)

                #################### train D using true_masks_a ##########################
                optimizer_D.zero_grad()
                
                real_patch = torch.cat([imgs, true_masks_a], dim=1)
                real_predict_D = net_D(real_patch)
                loss_adv_CE_real = L_adv_BCE(real_predict_D, real_labels)
                loss_adv_CE_real.backward()
                #########################
                
                masks_pred_D_A, masks_pred_D_fusion_A = net_G_A(imgs)
                
                masks_pred_D_softmax_A = F.softmax(masks_pred_D_A,dim=1)
                fake_patch_D = torch.cat([imgs, masks_pred_D_softmax_A], dim=1)
                fake_predict_D = net_D(fake_patch_D)
                #fake_predict_D_softmax = torch.sigmoid(fake_predict_D)
                loss_adv_CE_fake = L_adv_BCE(fake_predict_D, fake_labels)
                loss_adv_CE_fake.backward()

                D_Loss = (loss_adv_CE_real + loss_adv_CE_fake)
                epoch_loss_D += D_Loss.item()
                writer.add_scalar('Loss/D_train', D_Loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': D_Loss.item()})
                
                optimizer_D.step()


                #################### train D using true_masks_v ##########################
                optimizer_D.zero_grad()

                real_patch = torch.cat([imgs, true_masks_v], dim=1)
                real_predict_D = net_D(real_patch)
                loss_adv_CE_real = L_adv_BCE(real_predict_D, real_labels)
                loss_adv_CE_real.backward()
                #########################
                
                masks_pred_D_V, masks_pred_D_fusion_V = net_G_V(imgs)
                masks_pred_D_softmax_V = F.softmax(masks_pred_D_V,dim=1)
                fake_patch_D = torch.cat([imgs, masks_pred_D_softmax_V], dim=1)
                fake_predict_D_V = net_D(fake_patch_D)
                #fake_predict_D_softmax = F.softmax(fake_predict_D_V)
                loss_adv_CE_fake = L_adv_BCE(fake_predict_D_V, fake_labels)
                loss_adv_CE_fake.backward()

                D_Loss = (loss_adv_CE_real + loss_adv_CE_fake)
                epoch_loss_D += D_Loss.item()
                writer.add_scalar('Loss/D_train', D_Loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': D_Loss.item()})
                
                optimizer_D.step()


                #################### train D using true_masks##########################
                optimizer_D.zero_grad()
                real_patch = torch.cat([imgs, encode_tensor], dim=1)
                real_predict_D = net_D(real_patch)
                loss_adv_CE_real = L_adv_BCE(real_predict_D, real_labels)
                loss_adv_CE_real.backward()

                #########################
                masks_pred_D_fusion_A = masks_pred_D_fusion_A.detach()
                masks_pred_D_fusion_V = masks_pred_D_fusion_V.detach()

                masks_pred_D,_,_,_ = net_G(imgs, masks_pred_D_fusion_A, masks_pred_D_fusion_V)
                masks_pred_D_softmax = F.softmax(masks_pred_D,dim=1)
                fake_patch_D = torch.cat([imgs, masks_pred_D_softmax], dim=1)
                fake_predict_D = net_D(fake_patch_D)
                #fake_predict_D_softmax = F.softmax(fake_predict_D)
                loss_adv_CE_fake = L_adv_BCE(fake_predict_D, fake_labels)

                loss_adv_CE_fake.backward()
                D_Loss = (loss_adv_CE_real + loss_adv_CE_fake)

                epoch_loss_D += D_Loss.item()
                writer.add_scalar('Loss/D_train', D_Loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': D_Loss.item()})
                
                optimizer_D.step()
                

                ################### train G_A ###########################
                optimizer_G_A.zero_grad()
                masks_pred_G_A, masks_pred_D_fusion_A = net_G_A(imgs)
                masks_pred_G_softmax_A = F.softmax(masks_pred_G_A,dim=1)
                fake_patch_G = torch.cat([imgs, masks_pred_G_softmax_A], dim=1)
                fake_predict_G = net_D(fake_patch_G)
                #fake_predict_G_softmax = F.softmax(fake_predict_G)

                loss_adv_G_fake = L_adv_BCE(fake_predict_G, real_labels)
            
                #loss_seg_CE = L_seg_CE(masks_pred_G.flatten(start_dim=1, end_dim=3), true_masks.flatten(start_dim=1, end_dim=3))
                loss_seg_CE = L_seg_CE(masks_pred_G_A, true_masks)
                loss_seg_MSE = L_seg_MSE(masks_pred_G_softmax_A, encode_tensor)

                G_Loss = gama_hyper*loss_adv_G_fake + beta_hyper*loss_seg_CE + alpha_hyper*loss_seg_MSE 
                
                epoch_loss_G += G_Loss.item()
                writer.add_scalar('Loss/G_train_A', G_Loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': G_Loss.item()})
                G_Loss.backward()
                optimizer_G_A.step()

                ################### train G_V ###########################
                optimizer_G_V.zero_grad()
                masks_pred_G_V, masks_pred_D_fusion_V = net_G_V(imgs)
                masks_pred_G_softmax_V = F.softmax(masks_pred_G_V,dim=1)
                fake_patch_G = torch.cat([imgs, masks_pred_G_softmax_V], dim=1)
                fake_predict_G = net_D(fake_patch_G)
                #fake_predict_G_softmax = F.softmax(fake_predict_G)

                loss_adv_G_fake = L_adv_BCE(fake_predict_G, real_labels)
                #loss_seg_CE = L_seg_CE(masks_pred_G.flatten(start_dim=1, end_dim=3), true_masks.flatten(start_dim=1, end_dim=3))
                loss_seg_CE = L_seg_CE(masks_pred_G_V, true_masks)
                loss_seg_MSE = L_seg_MSE(masks_pred_G_softmax_V, encode_tensor)
                G_Loss = gama_hyper*loss_adv_G_fake + beta_hyper*loss_seg_CE + alpha_hyper*loss_seg_MSE 
                
                epoch_loss_G += G_Loss.item()
                writer.add_scalar('Loss/G_train_A', G_Loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': G_Loss.item()})
                G_Loss.backward()
                optimizer_G_V.step()

                ################### train G ###########################
                optimizer_G.zero_grad()
                masks_pred_G_A_part = masks_pred_D_fusion_A.detach()
                masks_pred_G_V_part = masks_pred_D_fusion_V.detach()
                masks_pred_G, side_1, side_2, side_3 = net_G(imgs, masks_pred_G_A_part, masks_pred_G_V_part)
                masks_pred_G_softmax = F.softmax(masks_pred_G,dim=1)
                side_1_softmax = F.softmax(side_1,dim=1)
                side_2_softmax = F.softmax(side_2,dim=1)
                side_3_softmax = F.softmax(side_3,dim=1)
                fake_patch_G = torch.cat([imgs, masks_pred_G_softmax], dim=1)
                fake_predict_G = net_D(fake_patch_G)
                #fake_predict_G_softmax = F.softmax(fake_predict_G)

                loss_adv_G_fake = L_adv_BCE(fake_predict_G, real_labels)
                #loss_seg_CE = L_seg_CE(masks_pred_G.flatten(start_dim=1, end_dim=3), true_masks.flatten(start_dim=1, end_dim=3))
                loss_seg_CE = L_seg_CE(masks_pred_G, true_masks)
                
                loss_seg_MSE = L_seg_MSE(masks_pred_G_softmax, encode_tensor)
                # S1 output
                loss_seg_CE_1 = L_seg_CE(side_1, true_masks)
                loss_seg_MSE_1 = L_seg_MSE(side_1_softmax, encode_tensor)
                # S2 output
                loss_seg_CE_2 = L_seg_CE(side_2, true_masks)
                loss_seg_MSE_2 = L_seg_MSE(side_2_softmax, encode_tensor)
                # S3 output
                loss_seg_CE_3 = L_seg_CE(side_3, true_masks)
                loss_seg_MSE_3 = L_seg_MSE(side_3_softmax, encode_tensor)

                G_Loss = gama_hyper*loss_adv_G_fake + beta_hyper*loss_seg_CE + alpha_hyper*loss_seg_MSE + 1/2*(alpha_hyper*loss_seg_MSE_1 + beta_hyper*loss_seg_CE_1) + \
                    1/4*(alpha_hyper*loss_seg_MSE_2 + beta_hyper*loss_seg_CE_2) + 1/8*(alpha_hyper*loss_seg_MSE_3 + beta_hyper*loss_seg_CE_3)
                

                epoch_loss_G += G_Loss.item()
                writer.add_scalar('Loss/G_train', G_Loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': G_Loss.item()})
                G_Loss.backward()
                optimizer_G.step()

                ##########################################################

                pbar.update(imgs.shape[0])
                global_step += 1
                with torch.no_grad():
                    if global_step % (n_train // ( batch_size)) == 0:
                    #if True:
                        acc, sensitivity, specificity, precision, G, F1_score_2, auc_roc, auc_pr, mse, iou,_ = eval_net(epoch, net_G, net_G_A, net_G_V, args.dataset, val_loader, device, mode='whole',train_or='train')[0:11]

                        scheduler_G.step(G_Loss.item())
                        scheduler_D.step(D_Loss.item())
                        writer.add_scalar('learning_rate', optimizer_G.param_groups[0]['lr'], global_step)
                        logging.info('Validation sensitivity: {}'.format(sensitivity))
                        writer.add_scalar('sensitivity/val_G', sensitivity, global_step)
                        logging.info('Validation specificity: {}'.format(specificity))
                        writer.add_scalar('specificity/val_G', specificity, global_step)
                        logging.info('Validation precision: {}'.format(precision))
                        writer.add_scalar('precision/val_G', precision, global_step)
                        logging.info('Validation G: {}'.format(G))
                        writer.add_scalar('G/val_G', G, global_step)
                        logging.info('Validation F1_score: {}'.format(F1_score_2))
                        writer.add_scalar('F1_score/val_G', F1_score_2, global_step)
                        logging.info('Validation mse: {}'.format(mse))
                        writer.add_scalar('mse/val_G', mse, global_step)
                        logging.info('Validation iou: {}'.format(iou))
                        writer.add_scalar('iou/val_G', iou, global_step)
                        logging.info('Validation acc: {}'.format(acc))
                        writer.add_scalar('Acc/val_G', acc, global_step)
                        logging.info('Validation auc_roc: {}'.format(auc_roc))
                        writer.add_scalar('Auc_roc/val_G', auc_roc, global_step)
                        logging.info('Validation auc_pr: {}'.format(auc_pr))
                        writer.add_scalar('Auc_pr/val_G', auc_pr, global_step)

                        prediction_binary = (F.softmax(masks_pred_G,dim=1) > 0.5)
                        prediction_binary_gpu = prediction_binary.to(device=device, dtype=mask_type)
                        total_train_pixel += prediction_binary_gpu.nelement()
                        real_predict_binary = (F.softmax(real_predict_D,dim=1) > 0.5)
                        real_predict_binary_gpu = real_predict_binary.to(device=device, dtype=mask_type)
                        fake_predict_binary = (F.softmax(fake_predict_D,dim=1) > 0.5)
                        fake_predict_binary_gpu = fake_predict_binary.to(device=device, dtype=mask_type)
                        prediction_binary_DR = real_predict_binary_gpu.eq(real_labels.data).sum().item()
                        prediction_binary_DF = fake_predict_binary_gpu.eq(fake_labels.data).sum().item()
                        aver_prediction_binary_D = (prediction_binary_DR + prediction_binary_DF)/2
                        train_accuracy_D = 100 * aver_prediction_binary_D / total_train_pixel
                        logging.info('Validation accuracy: {}'.format(train_accuracy_D))
                        writer.add_scalar('Acc/Val_D', train_accuracy_D, global_step)

                        total_train_pixel = 0

            if F1_score_2 > best_F1:
                best_F1=F1_score_2
                if save_cp:
                    try:
                        os.mkdir(dir_checkpoint)
                        logging.info('Created checkpoint directory')
                    except OSError:
                        pass
                    torch.save(net_G.state_dict(),
                            dir_checkpoint + f'CP_best_F1_all.pth')
                    torch.save(net_G_A.state_dict(),
                            dir_checkpoint + f'CP_best_F1_A.pth')
                    torch.save(net_G_V.state_dict(),
                            dir_checkpoint + f'CP_best_F1_V.pth')
                    logging.info(f'Checkpoint {epoch + 1} saved !')


    writer.close()



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=240, help='Number of epochs', dest='epochs')
    parser.add_argument('--batch-size', type=int, default=6, help='Batch size', dest='batchsize')
    parser.add_argument('--learning-rate', type=float, nargs='?', default=2e-4, help='Learning rate', dest='lr')
    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file', dest='load')
    parser.add_argument('--job_name', type=str, default='J', help='type of discriminator', dest='jn')
    parser.add_argument('--discriminator', type=str, default=False, help='type of discriminator', dest='dis')
    parser.add_argument('--dataset', type=str, help='dataset name', dest='dataset')
    parser.add_argument('--validation', type=float, default=5.0, help='Percent of the data validation', dest='val')
    parser.add_argument('--uniform', type=str, default='False', help='whether to uniform the image size', dest='uniform')
    parser.add_argument('--seed_num', type=int, default=42, help='Validation split seed', dest='seed')
    parser.add_argument('--alpha', dest='alpha', type=float, help='alpha')
    parser.add_argument('--beta', dest='beta', type=float, help='beta')
    parser.add_argument('--gama', dest='gama', type=float, help='gama')

    return parser.parse_args()


if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    img_size = Define_image_size(args.uniform, args.dataset)

    net_G = Generator_main(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_D = Discriminator(input_channels=7, n_filters = 32, n_classes=4, bilinear=False)
    net_G_A = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
    net_G_V = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)

    logging.info(f'Network_G:\n'
                 f'\t{net_G.n_channels} input channels\n'
                 f'\t{net_G.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net_G.bilinear else "Transposed conv"} upscaling')

    logging.info(f'Network_D:\n'
                 f'\t{net_D.n_channels} input channels\n'
                 f'\t{net_D.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net_D.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net_G.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
        net_D.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net_G.to(device=device)
    net_D.to(device=device)
    net_G_A.to(device=device)
    net_G_V.to(device=device)


    train_net(net_G=net_G,
                net_D=net_D,
                net_G_A=net_G_A,
                net_G_V=net_G_V,
                epochs=args.epochs,
                batch_size=args.batchsize,
                alpha_hyper=args.alpha,
                beta_hyper=args.beta,
                gama_hyper=args.gama,
                lr=args.lr,
                device=device,
                val_percent=args.val / 100,
                image_size=img_size)
    

