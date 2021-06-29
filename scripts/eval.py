import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torch.autograd import Function
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from skimage import filters
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import os


def pixel_values_in_mask(true_vessels, pred_vessels, mask, train_or, dataset):

    if train_or=='val':
        true_vessels = np.squeeze(true_vessels)
        pred_vessels = np.squeeze(pred_vessels)

        if dataset=='HRF-AV':
            true_vessels = (true_vessels[mask[0,...] != 0])
            pred_vessels = (pred_vessels[mask[0,...] != 0])
        else:
            true_vessels = (true_vessels[mask!= 0])
            pred_vessels = (pred_vessels[mask!= 0])
        
        assert np.max(pred_vessels)<=1.0 and np.min(pred_vessels)>=0.0
        assert np.max(true_vessels)==1.0 and np.min(true_vessels)==0.0

    return true_vessels.flatten(), pred_vessels.flatten()

def AUC_ROC(true_vessel_arr, pred_vessel_arr):

    AUC_ROC=roc_auc_score(true_vessel_arr, pred_vessel_arr)
    return AUC_ROC

def threshold_by_otsu(pred_vessels):
    
    threshold=filters.threshold_otsu(pred_vessels)
    pred_vessels_bin=np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels>=threshold]=1

    return pred_vessels_bin

def AUC_PR(true_vessel_img, pred_vessel_img):

    precision, recall, _ = precision_recall_curve(true_vessel_img.flatten(), pred_vessel_img.flatten(),  pos_label=1)
    AUC_prec_rec = auc(recall, precision)
    return AUC_prec_rec

def misc_measures(true_vessel_arr, pred_vessel_arr):
    cm=confusion_matrix(true_vessel_arr, pred_vessel_arr)
    mse = mean_squared_error(true_vessel_arr, pred_vessel_arr)

    try:
        acc=1.*(cm[0,0]+cm[1,1])/np.sum(cm)
        sensitivity=1.*cm[1,1]/(cm[1,0]+cm[1,1])
        specificity=1.*cm[0,0]/(cm[0,1]+cm[0,0])
        precision=1.*cm[1,1]/(cm[1,1]+cm[0,1])
        G = np.sqrt(sensitivity*specificity)
        F1_score_2 = 2*precision*sensitivity/(precision+sensitivity)
        iou = 1.*cm[1,1]/(cm[1,0]+cm[0,1]+cm[1,1])
        return acc, sensitivity, specificity, precision, G, F1_score_2, mse, iou
    
    except:

        return 0,0,0,0,0,0,0,0


def eval_net(epoch, net, net_a, net_v, dataset, loader, device, mode, train_or):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    
    print(dataset)
    if dataset=='HRF-AV':
        image_size = (3504, 2336)
    elif dataset=='LES-AV':
        image_size = (1620,1444)
    else:
        image_size = (592,592)        
    mask_type = torch.float32 if net.n_classes == 1 else torch.float32
    n_val = len(loader) 
    acc_a,sent_a,spet_a,pret_a,G_t_a,F1t_a,mset_a,iout_a,auc_roct_a,auc_prt_a=0,0,0,0,0,0,0,0,0,0
    acc_v,sent_v,spet_v,pret_v,G_t_v,F1t_v,mset_v,iout_v,auc_roct_v,auc_prt_v=0,0,0,0,0,0,0,0,0,0
    acc_u,sent_u,spet_u,pret_u,G_t_u,F1t_u,mset_u,iout_u,auc_roct_u,auc_prt_u=0,0,0,0,0,0,0,0,0,0
    acc,sent,spet,pret,G_t,F1t,mset,iout,auc_roct,auc_prt=0,0,0,0,0,0,0,0,0,0

    num = 0
    
    seg_results_small_path = dataset + '/Final_pre/small_pre/'
    seg_results_raw_path = dataset + '/Final_pre/raw_pre/'
    
    if not os.path.isdir(seg_results_small_path):
        os.makedirs(seg_results_small_path)

    if not os.path.isdir(seg_results_raw_path):
        os.makedirs(seg_results_raw_path)
        
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, label, mask = batch['image'], batch['label'], batch['mask']
            img_name = batch['name'][0]
            
            imgs = imgs.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=mask_type)

            with torch.no_grad():

                num +=1
                masks_pred_G = net_a(imgs)
                masks_pred_G_sigmoid_A = torch.sigmoid(masks_pred_G)
                
                masks_pred_G = net_v(imgs)
                masks_pred_G_sigmoid_V = torch.sigmoid(masks_pred_G)

                masks_pred_G_sigmoid_A_part = masks_pred_G_sigmoid_A.detach()
                masks_pred_G_sigmoid_V_part = masks_pred_G_sigmoid_V.detach()

                mask_pred,_,_,_ = net(imgs, masks_pred_G_sigmoid_A_part, masks_pred_G_sigmoid_V_part)

                mask_pred_tensor_small = mask_pred.clone().detach()
                mask_pred_tensor_small = torch.sigmoid(mask_pred_tensor_small)
                mask_pred_tensor_small = torch.squeeze(mask_pred_tensor_small)

                if train_or=='val':
                
                    save_image(mask_pred_tensor_small, seg_results_small_path+ img_name+ '.png')
                    if dataset!='DRIVE_AV':
                        mask_pred_img = Image.open(seg_results_small_path+ img_name + '.png').resize((image_size)) 
                    else:
                        mask_pred_img = Image.open(seg_results_small_path+ img_name + '.png')           
                    mask_pred = torchvision.transforms.ToTensor()(mask_pred_img)
                    mask_pred = torch.unsqueeze(mask_pred, 0)
                    
                    if dataset!='HRF-AV':
                        mask_pred[mask.repeat(1, 3, 1, 1)== 0]=0
                    else:
                        mask_pred[mask == 0]=0                        
                    save_image(mask_pred, seg_results_raw_path + img_name+ '.png')
            

            if mode== 'whole':
                ########################################

                # based on the whole images

                ########################################
                mask_pred_sigmoid = mask_pred
                mask_pred_sigmoid_cpu = mask_pred_sigmoid.detach().cpu().numpy()
                mask_pred_sigmoid_cpu = np.squeeze(mask_pred_sigmoid_cpu)

                label_cpu = label.detach().cpu().numpy()
                label_cpu = np.squeeze(label_cpu)

                mask_cpu = mask.detach().cpu().numpy()
                mask_cpu = np.squeeze(mask_cpu)

                label_cpu = label_cpu.transpose((1, 2, 0))
                mask_pred_sigmoid_cpu = mask_pred_sigmoid_cpu.transpose((1, 2, 0))
                binarys_in_mask_vessel=((mask_pred_sigmoid_cpu)>0.5).astype('float')

                encoded_pred_a = np.zeros(binarys_in_mask_vessel.shape[0:2], dtype=int)
                encoded_pred_u = np.zeros(binarys_in_mask_vessel.shape[0:2], dtype=int)
                encoded_pred_v = np.zeros(binarys_in_mask_vessel.shape[0:2], dtype=int)

                encoded_gt_a = np.zeros(label_cpu.shape[0:2], dtype=int)
                encoded_gt_u = np.zeros(label_cpu.shape[0:2], dtype=int)
                encoded_gt_v = np.zeros(label_cpu.shape[0:2], dtype=int)

                white_ind = np.where(np.logical_and(label_cpu[:,:,0] == 1, label_cpu[:,:,1] == 1, label_cpu[:,:,2] == 1))
                if white_ind[0].size != 0:
                    label_cpu[white_ind] = [0,1,0]

                white_ind_pre = np.where(np.logical_and(binarys_in_mask_vessel[:,:,0] == 1, binarys_in_mask_vessel[:,:,1] == 1, binarys_in_mask_vessel[:,:,2] == 1))
                if white_ind_pre[0].size != 0:
                    binarys_in_mask_vessel[white_ind_pre] = [0,1,0]
                
                arteriole = np.where(np.logical_and(label_cpu[:,:,0] == 1, label_cpu[:,:,1] == 0)); encoded_gt_a[arteriole] = 1
                venule = np.where(np.logical_and(label_cpu[:,:,2] == 1, label_cpu[:,:,1] == 0)); encoded_gt_v[venule] = 1
                uncertainty = np.where(np.logical_and(label_cpu[:,:,1] == 1, label_cpu[:,:, 0] == 0, label_cpu[:,:, 2] == 0)); encoded_gt_u[uncertainty] = 1
                arteriole = np.where(np.logical_and(binarys_in_mask_vessel[:,:,0] == 1, binarys_in_mask_vessel[:,:,1] == 0)); encoded_pred_a[arteriole] = 1
                venule = np.where(np.logical_and(binarys_in_mask_vessel[:,:,2] == 1, binarys_in_mask_vessel[:,:,1] == 0)); encoded_pred_v[venule] = 1
                uncertainty = np.where(np.logical_and(binarys_in_mask_vessel[:,:,1] == 1,binarys_in_mask_vessel[:,:, 0] == 0, binarys_in_mask_vessel[:,:, 2] == 0)); encoded_pred_u[uncertainty] = 1

                count_artery = np.sum(encoded_gt_a==1)
                count_vein = np.sum(encoded_gt_v==1)
                count_uncertainty = np.sum(encoded_gt_u==1)
                count_total = count_artery + count_vein + count_uncertainty

                ##########################################
                #artery
                #######################################
                encoded_gt_vessel_point_a, encoded_pred_vessel_point_a = pixel_values_in_mask(encoded_gt_a, encoded_pred_a, mask_cpu, train_or, dataset)

                auc_roc_a=AUC_ROC(encoded_gt_vessel_point_a,encoded_pred_vessel_point_a)
                auc_pr_a=AUC_PR(encoded_gt_vessel_point_a, encoded_pred_vessel_point_a)

                acc_ve_a, sensitivity_ve_a, specificity_ve_a, precision_ve_a, G_ve_a, F1_score_ve_a, mse_a, iou_a = misc_measures(encoded_gt_vessel_point_a, encoded_pred_vessel_point_a)
        
                acc_a+=acc_ve_a
                sent_a+=sensitivity_ve_a
                spet_a+=specificity_ve_a
                pret_a+=precision_ve_a
                G_t_a+=G_ve_a
                F1t_a+=F1_score_ve_a
                mset_a+=mse_a
                iout_a+=iou_a
                auc_roct_a+=auc_roc_a
                auc_prt_a+=auc_pr_a

                
                ##########################################
                #vein
                #######################################
                encoded_gt_vessel_point_v, encoded_pred_vessel_point_v = pixel_values_in_mask(encoded_gt_v, encoded_pred_v, mask_cpu, train_or, dataset)

                auc_roc_v=AUC_ROC(encoded_gt_vessel_point_v,encoded_pred_vessel_point_v)
                auc_pr_v=AUC_PR(encoded_gt_vessel_point_v, encoded_pred_vessel_point_v)

                acc_ve_v, sensitivity_ve_v, specificity_ve_v, precision_ve_v, G_ve_v, F1_score_ve_v, mse_v, iou_v = misc_measures(encoded_gt_vessel_point_v, encoded_pred_vessel_point_v)
        
                acc_v+=acc_ve_v
                sent_v+=sensitivity_ve_v
                spet_v+=specificity_ve_v
                pret_v+=precision_ve_v
                G_t_v+=G_ve_v
                F1t_v+=F1_score_ve_v
                mset_v+=mse_v
                iout_v+=iou_v
                auc_roct_v+=auc_roc_v
                auc_prt_v+=auc_pr_v

                
                ##########################################
                #uncertainty
                #######################################
                encoded_gt_vessel_point_u, encoded_pred_vessel_point_u = pixel_values_in_mask(encoded_gt_u, encoded_pred_u, mask_cpu,train_or, dataset)

                auc_roc_u=AUC_ROC(encoded_gt_vessel_point_u,encoded_pred_vessel_point_u)
                auc_pr_u=AUC_PR(encoded_gt_vessel_point_u, encoded_pred_vessel_point_u)

                acc_ve_u, sensitivity_ve_u, specificity_ve_u, precision_ve_u, G_ve_u, F1_score_ve_u, mse_u, iou_u = misc_measures(encoded_gt_vessel_point_u, encoded_pred_vessel_point_u)
        
                
                if np.isnan(F1_score_ve_u):
                    acc_ve_u = 0
                    sensitivity_ve_u = 0
                    specificity_ve_u = 0
                    precision_ve_u = 0
                    G_ve_u = 0
                    F1_score_ve_u = 0
                    mse_u = 0
                    iou_u = 0
                    auc_roc_u = 0
                    auc_pr_u = 0
                    

                acc_u+=acc_ve_u
                sent_u+=sensitivity_ve_u
                spet_u+=specificity_ve_u
                pret_u+=precision_ve_u
                G_t_u+=G_ve_u
                F1t_u+=F1_score_ve_u
                mset_u+=mse_u
                iout_u+=iou_u
                auc_roct_u+=auc_roc_u
                auc_prt_u+=auc_pr_u

                acc+=(count_artery*acc_ve_a + count_vein*acc_ve_v + count_uncertainty*acc_ve_u)/count_total
                sent+=(count_artery*sensitivity_ve_a + count_vein*sensitivity_ve_v + count_uncertainty*sensitivity_ve_u)/count_total
                spet+=(count_artery*specificity_ve_a + count_vein*specificity_ve_v + count_uncertainty*specificity_ve_u)/count_total
                pret+=(count_artery*precision_ve_a + count_vein*precision_ve_v + count_uncertainty*precision_ve_u)/count_total
                G_t+=(count_artery*G_ve_a + count_vein*G_ve_v + count_uncertainty*G_ve_u)/count_total
                F1t+=(count_artery*F1_score_ve_a + count_vein*F1_score_ve_v + count_uncertainty*F1_score_ve_u)/count_total
                mset+=(count_artery*mse_a + count_vein*mse_v + count_uncertainty*mse_u)/count_total
                iout+=(count_artery*iou_a + count_vein*iou_v + count_uncertainty*iou_u)/count_total
                auc_roct+=(count_artery*auc_roc_a + count_vein*auc_roc_v + count_uncertainty*auc_roc_u)/count_total
                auc_prt+=(count_artery*auc_pr_a + count_vein*auc_pr_v + count_uncertainty*auc_pr_u)/count_total


    net.train()
    
    return  acc/ n_val, sent/ n_val, spet/ n_val, pret/ n_val, G_t/ n_val, F1t/ n_val, auc_roct/ n_val, auc_prt/ n_val, mset/ n_val, iout/ n_val, \
        acc_a/ n_val, sent_a/ n_val, spet_a/ n_val, pret_a/ n_val, G_t_a/ n_val, F1t_a/ n_val, auc_roct_a/ n_val, auc_prt_a/ n_val, mset_a/ n_val, iout_a/ n_val, \
            acc_v/ n_val, sent_v/ n_val, spet_v/ n_val, pret_v/ n_val, G_t_v/ n_val, F1t_v/ n_val, auc_roct_v/ n_val, auc_prt_v/ n_val, mset_v/ n_val, iout_v/ n_val, \
                acc_u/ n_val, sent_u/ n_val, spet_u/ n_val, pret_u/ n_val, G_t_u/ n_val, F1t_u/ n_val, auc_roct_u/ n_val, auc_prt_u/ n_val, mset_u/ n_val, iout_u/ n_val


