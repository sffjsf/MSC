import sys
import os
import time
import random
import importlib

#from thop import profile
from progress.bar import Bar
from collections import OrderedDict
from util import *
from PIL import Image
from data_my import Train_Dataset, Test_Dataset
from test import test_model
import torch
from torch.nn import utils
from base.framework_factory import load_framework
import cv2
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('log')
criterion = nn.BCELoss()
criterion1 = nn.BCELoss(reduction='none')
torch.set_printoptions(precision=5)
import math
import torch
import torch.nn as nn


import cv2
import numpy as np


# PNC
class PNCModule:
    def __init__(self, num_classes=2, device='cuda'):
        self.num_classes = num_classes
        self.device = device
        self.average_predictions = {}
        self.update_counts = {}
        self.selection_matrices = {}
        
    def update_selection_matrix(self, image_ids, selection_matrix):
        """Update the selection matrix"""
        for i, img_id in enumerate(image_ids):
            self.selection_matrices[img_id] = selection_matrix[i].detach().clone().to(self.device)
     

    def update_prediction_history(self, image_ids, predictions):
    
        for i, img_id in enumerate(image_ids):
            current_prediction = predictions[i].detach().clone().to(self.device)
            
            if img_id not in self.average_predictions:
                # First Update: Use the current prediction directly
                self.average_predictions[img_id] = current_prediction
                self.update_counts[img_id] = 1  
            else:
                prev_avg = self.average_predictions[img_id]
                prev_denom = self.update_counts[img_id]
                alpha = math.exp(-0.5)  # Calculate attenuation factor
                new_denom = alpha * prev_denom + 1
                new_avg = (alpha * prev_denom * prev_avg + current_prediction) / new_denom
                # Update the stored value            
                self.average_predictions[img_id] = new_avg
                self.update_counts[img_id] = new_denom
   
            
    def calculate_average_prediction(self, image_ids):
        """计算平均预测"""
        batch_size = len(image_ids)
        
        # Obtain the average predicted shape of the first image as a reference
        if image_ids and image_ids[0] in self.average_predictions:
            example_pred = self.average_predictions[image_ids[0]]
            C, H, W = example_pred.shape
        else:
            # If there is no historical data, return a zero tensor
            C, H, W = 1, 256, 256  # Default shape, which can be adjusted according to actual circumstances
            
        avg_predictions = torch.zeros(batch_size, C, H, W).to(self.device)
        
        for i, img_id in enumerate(image_ids):
            if img_id in self.average_predictions:
                avg_predictions[i] = self.average_predictions[img_id]
            # If there is no historical data, keep it at zero (this situation should not occur as the update_prediction_history function will be called first)
                
        return avg_predictions
    
    def calculate_epsilon(self, avg_predictions):
        """计算权衡权重ε"""
        batch_size, C, H, W = avg_predictions.shape
        # Positive Probability
        pos_prob = torch.clamp(avg_predictions, 1e-8, 1-1e-8)
        # Negative Probability
        neg_prob = 1 - pos_prob
        # Merged into a complete probability distribution with shape (B, 2, H, W)
        prob_distribution = torch.cat([neg_prob, pos_prob], dim=1)
        # Calculate the average entropy H(avg_s_n^m)
        # For each pixel, calculate the entropy of the two categories: -Σ p * log(p)
        entropy = -torch.sum(prob_distribution * torch.log(prob_distribution + 1e-8), dim=1, keepdim=True)
        # Entropy of uniform distribution - ln(u) = -ln(1/2) = ln(2)
        uniform_entropy = torch.log(torch.tensor(2.0, dtype=torch.float)).to(self.device)
        epsilon = 1 - entropy / uniform_entropy
        
        return epsilon
    
    def correct_labels(self, image_ids, original_labels, avg_predictions, epsilon):
        """校正噪声标签"""
        batch_size, C, H, W = original_labels.shape
        corrected_labels = torch.zeros_like(original_labels)
        
        for i, img_id in enumerate(image_ids):
            if img_id in self.selection_matrices:
                O = self.selection_matrices[img_id]
                O_complement = 1 - O
                corrected_outliers = O_complement * ((1 - epsilon[i]) * original_labels[i] + epsilon[i] * avg_predictions[i])
                corrected_inliers = O * original_labels[i]
                corrected_labels[i] = corrected_outliers + corrected_inliers
                corrected_labels[i] = torch.clamp(corrected_outliers + corrected_inliers, 0, 1)
            else:
                corrected_labels[i] = original_labels[i]
                
        return corrected_labels
    
def compute_Sx(cur_it, total_it):
    """
    Calculate the value of S_x, where a increases linearly with cur_it. 
    Parameters:
    cur_it: Current iteration count
    total_it: Total number of iterations
    p: Probability tensor
    eps: Small value to prevent log(0), default is 1e-8 
    Return:
    S_x: The resulting tensor obtained through calculation
    """
    # # Calculate the value of 'a': It increases linearly from 0.2 to 1. When 'cur_it' reaches 0.5 * 'total_it', 'a' becomes 1, and thereafter it remains at 1.
    a = 0.1 + (1.8 / total_it) * cur_it  
    a = min(a, 1.0)  
    return a
def paper_osl_loss(predictions, noisy_labels,noisy_labels1,noisy_labels2,cur_it,total_it):
    """
    Args:
    predictions (torch.Tensor): Probability map of model predictions, with shape (B, 1, H, W) and values within the range [0, 1].
    noisy_labels (torch.Tensor): Noisy labels, with the same shape as predictions. 
    Returns:
    torch.Tensor: The calculated loss value.
    """
    a=compute_Sx(cur_it,total_it)
    if a>=1.1:
        b=1000
    else:
        b=5
    
    noisy_labels=noisy_labels.gt(0.5).float()
    noisy_labels1=noisy_labels1.gt(0.5).float()
    noisy_labels2=noisy_labels2.gt(0.5).float()
  
    loss_per_pixel = criterion1(predictions, noisy_labels) # Shape: (B, 1, H, W)
    loss_per_pixel1 = criterion1(predictions, noisy_labels1) # Shape: (B, 1, H, W)
    loss_per_pixel2 = criterion1(predictions, noisy_labels2) # Shape: (B, 1, H, W)
    # -------------------------------------------------------------------------
    # New Section: Entropy-Driven Adaptive Fusion
    # ------------------------------------------------------------------------- 
    # A. Obtain the basic result of Min-Loss (y_opt)
    # Stack the three Losses together
    combined_loss = torch.stack([loss_per_pixel, loss_per_pixel1, loss_per_pixel2], dim=1) # (B, 3, H, W)
    min_loss, min_indices = torch.min(combined_loss, dim=1) # (B, 1, H, W) - 注意 keepdim=True 方便 gather
    min_indices = min_indices.unsqueeze(1) # 确保维度为 (B, 1, H, W)
    all_noisy_labels = torch.stack([noisy_labels, noisy_labels1, noisy_labels2], dim=1) # (B, 3, H, W)
    y_opt = torch.gather(all_noisy_labels, 1, min_indices).squeeze(1) # (B, 1, H, W)
    count_of_ones = all_noisy_labels.sum(dim=1)
    y_1 = (count_of_ones >= 2.0).float()
    probs = torch.clamp(predictions, 1e-7, 1.0 - 1e-7) 
    entropy = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)
    
    max_entropy = torch.log(torch.tensor(2.0)).to(predictions.device)
    w = 1.0 - (entropy / max_entropy)
    w = torch.clamp(w, 0.0, 1.0) 
    y_sam = noisy_labels1 

    optimal_noisy_labels = w * y_opt + (1.0 - w) * y_1
    optimal_noisy_labels=optimal_noisy_labels.gt(0.5).float()

    loss_optimal = criterion1(predictions, optimal_noisy_labels)
    eps = 1e-8
    p = predictions
    
    S_x = -0.5 * (torch.log(p + eps) + torch.log(1 - p + eps))*a
    O_t00 = (loss_optimal <= S_x/a).float() # Shape: (B, 1, H, W)
 
    kernel_size = 5
    padding = kernel_size // 2
    sigmoid_k = 20.0  
    sigmoid_theta = 0.5  
    min_fg_pixels = 1  
    min_bg_pixels = 1  
    eps = 1e-8  

# -------------------------- Core Logic: Dual-end Neighborhood Confidence Optimization for Foreground and Background --------------------------
# 1. Extract foreground/background masks (covering all pixels, addressing the issue in the original code that only focused on the foreground)
    foreground_mask = (optimal_noisy_labels == 1.0).float()  
    background_mask = (optimal_noisy_labels == 0.0).float()  
    neighbor_fg_avg = F.avg_pool2d(foreground_mask, kernel_size=kernel_size, stride=1, padding=padding)
    neighbor_fg_count = neighbor_fg_avg * (kernel_size * kernel_size)  
   
    fg_reliable_mask = O_t00 * foreground_mask 
    neighbor_fg_reliable_avg = F.avg_pool2d(fg_reliable_mask, kernel_size=kernel_size, stride=1, padding=padding)
    neighbor_fg_reliable_count = neighbor_fg_reliable_avg * (kernel_size * kernel_size)  
    
    fg_neighbor_conf = torch.where(
        neighbor_fg_count >= min_fg_pixels,
        neighbor_fg_reliable_count / (neighbor_fg_count + eps),
        torch.zeros_like(neighbor_fg_count)
    )

   
    neighbor_bg_avg = F.avg_pool2d(background_mask, kernel_size=kernel_size, stride=1, padding=padding)
    neighbor_bg_count = neighbor_bg_avg * (kernel_size * kernel_size)  
   
    bg_reliable_mask = O_t00 * background_mask  
    neighbor_bg_reliable_avg = F.avg_pool2d(bg_reliable_mask, kernel_size=kernel_size, stride=1, padding=padding)
    neighbor_bg_reliable_count = neighbor_bg_reliable_avg * (kernel_size * kernel_size)  
    
    bg_neighbor_conf = torch.where(
        neighbor_bg_count >= min_bg_pixels,
        neighbor_bg_reliable_count / (neighbor_bg_count + eps),
        torch.zeros_like(neighbor_bg_count)
    )

    neighbor_confidence = torch.where(
        foreground_mask == 1.0,  
        fg_neighbor_conf,
        bg_neighbor_conf
    )

    
    O_t_neighbor = torch.sigmoid(sigmoid_k * (neighbor_confidence - sigmoid_theta))  
    O_t = torch.max(O_t00, O_t_neighbor)  
    O_t001=(loss_per_pixel1 <= S_x / a).float()

    
    foreground_mask = (noisy_labels1 == 1.0).float()  
    background_mask = (noisy_labels1 == 0.0).float()  

    neighbor_fg_avg = F.avg_pool2d(foreground_mask, kernel_size=kernel_size, stride=1, padding=padding)
    neighbor_fg_count = neighbor_fg_avg * (kernel_size * kernel_size)  
    
    fg_reliable_mask = O_t001 * foreground_mask  
    neighbor_fg_reliable_avg = F.avg_pool2d(fg_reliable_mask, kernel_size=kernel_size, stride=1, padding=padding)
    neighbor_fg_reliable_count = neighbor_fg_reliable_avg * (kernel_size * kernel_size)  
    
    fg_neighbor_conf = torch.where(
        neighbor_fg_count >= min_fg_pixels,
        neighbor_fg_reliable_count / (neighbor_fg_count + eps),
        torch.zeros_like(neighbor_fg_count)
    )

   
    neighbor_bg_avg = F.avg_pool2d(background_mask, kernel_size=kernel_size, stride=1, padding=padding)
    neighbor_bg_count = neighbor_bg_avg * (kernel_size * kernel_size) 
    
    bg_reliable_mask = O_t001 * background_mask  
    neighbor_bg_reliable_avg = F.avg_pool2d(bg_reliable_mask, kernel_size=kernel_size, stride=1, padding=padding)
    neighbor_bg_reliable_count = neighbor_bg_reliable_avg * (kernel_size * kernel_size)  
    
    bg_neighbor_conf = torch.where(
        neighbor_bg_count >= min_bg_pixels,
        neighbor_bg_reliable_count / (neighbor_bg_count + eps),
        torch.zeros_like(neighbor_bg_count)
    )

    
    neighbor_confidence = torch.where(
        foreground_mask == 1.0,  
        fg_neighbor_conf,
        bg_neighbor_conf
    )

    
    O_t_neighbor = torch.sigmoid(sigmoid_k * (neighbor_confidence - sigmoid_theta))  
    O_t1f = torch.max(O_t001, O_t_neighbor)  
   
    O_t1 = ((noisy_labels1 >= 0.5) * O_t1f).float()

    O_t0 = (loss_optimal <= S_x/a).float() 
    if cur_it>500:
        loss_optimal111=criterion1(predictions, noisy_labels1)
        
        numerator111 = torch.sum(loss_optimal111 * O_t1)
        
        denominator111 = torch.sum(O_t1) + eps
        
        total_loss = numerator111 / denominator111/3

        
        numerator = torch.sum(loss_optimal * O_t)
        
        denominator = torch.sum(O_t) + eps
        
        total_loss += numerator / denominator
    else:
        loss_optimal111=criterion1(predictions, noisy_labels1)
        total_loss=loss_optimal111.mean()/3
   
    return total_loss,optimal_noisy_labels, O_t0
 


def rgb2grey(images):
    mean = np.array([0.447, 0.407, 0.386])
    std = np.array([0.244, 0.250, 0.253])

    images1, images2, images3 = images[:, 0:1, :, :], images[:, 1:2, :, :], images[:, 2:3, :, :]
    images1 = images1 * std[0] + mean[0]
    images2 = images2 * std[1] + mean[1]
    images3 = images3 * std[2] + mean[2]
    img_grey = images1 * 0.299 + images2 * 0.587 + images3 * 0.114
    return img_grey
# --------------------------
# 3. Pseudo-boundary Label Generation Function (Paper: Generating Y^pe from the Salient Map S)
# --------------------------
def generate_pseudo_boundary(S, threshold=0.5):
    """
    Generate pseudo boundary labels Y^pe from the saliency map (the paper uses an edge detector; here, the Canny algorithm is employed) Args:
    S: The saliency map output by the model (batch, 1, H, W), pixel values ∈ [0, 1]
    threshold: The binarization threshold for the saliency map (to distinguish foreground/background) Returns:
    Y_pe: Pseudo boundary label (batch, 1, H, W), pixel values are 0/1 (1 indicates the boundary)
    """
    Y_pe = []
   
    for s in S:
       
        s_np = (s.squeeze().cpu().detach().numpy() * 255).astype(np.uint8)
        _, binary = cv2.threshold(s_np, threshold*255, 255, cv2.THRESH_BINARY)
        
        canny_edge = cv2.Canny(binary, 30, 100)  
        
        edge_tensor = torch.tensor(canny_edge / 255, dtype=torch.float32).unsqueeze(0)
        Y_pe.append(edge_tensor)
    
    Y_pe = torch.stack(Y_pe, dim=0).to(S.device)
    
    return Y_pe.detach()

# --------------------------
# 4. 边界损失计算函数（论文公式2：BCE损失）
# --------------------------
def compute_bdy_loss(S_c, Y_pe):
    """
    Calculate the boundary loss L_bdy (binary cross-entropy loss) Args:
    S_c: The predicted boundary map output by EPM (batch, 1, H, W)
    Y_pe: Pseudo boundary label (batch, 1, H, W) Returns:
    L_bdy: Scalar loss value
    """
    loss_bdy = criterion(S_c, Y_pe)
    return loss_bdy

# --------------------------
def main():
    
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return
    pnc_module = PNCModule(num_classes=2, device='cuda')
    # Loading model
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    config['stage'] = 2

    # Loading datasets
    train_loader = Train_Dataset(config) # get_loader(config)
    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)
    
    debug = config['debug']
    num_epoch = config['epoch']
    num_iter = train_loader.size
    ave_batch = config['ave_batch']
    trset = config['trset']
    batch_idx = 0
    model.zero_grad()
    for epoch in range(1, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()
        
        if debug:
            test_model(model, test_sets, config, epoch)

        st = time.time()
        loss_count = 0
        crf_count = 0
        optim.zero_grad()
        sche.step()
        iter_per_epoch = num_iter // config['batch']
        index_list = np.array(range(num_iter))
        random.shuffle(index_list)
        index_list = index_list[:iter_per_epoch * config['batch']]
        index_list = np.array(index_list).reshape((iter_per_epoch, config['batch']))
        
        print('Current LR: {:.6f}.'.format(optim.param_groups[1]['lr']))
        bar = Bar('{:10}-{:8} | epoch {:2}:'.format(net_name, config['sub'], epoch), max=iter_per_epoch)

        lamda = config['resdual']

        for i, idx_list in enumerate(index_list):

            cur_it = i + (epoch-1) * iter_per_epoch
            total_it = num_epoch * iter_per_epoch
            
           
            images, gts, gts1,gts2= train_loader.images[idx_list], train_loader.gts[idx_list], train_loader.gts1[idx_list], train_loader.gts2[idx_list]

            images = torch.tensor(np.array(images)).float().cuda()
            gts = torch.tensor( gts).float().cuda()
            gts1 = torch.tensor(gts1).float().cuda()
            gts2 = torch.tensor(gts2).float().cuda()
            
            if config['multi']:
                scales = [-1, 0, 1] 
                #scales = [-2, -1, 0, 1, 2] 
                input_size = config['size']
                input_size += int(np.random.choice(scales, 1) * 64)
                images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')
            
            Y = model(images, 'train')
            image_ids = [f"img_{idx}" for idx in idx_list]
            pnc_module.update_prediction_history(image_ids, torch.sigmoid(Y['final']))
            loss_paper, selected_gts, selection_matrix = paper_osl_loss(torch.sigmoid(Y['final']), gts, gts1, gts2,cur_it,total_it)
            
            
            
            pnc_module.update_selection_matrix(image_ids, selection_matrix)
            
            loss = model_loss(Y, selected_gts, config) / ave_batch
            loss+=2*loss_paper
            Y_c=generate_pseudo_boundary(torch.sigmoid(Y['final']))
            loss1 =compute_bdy_loss(Y['bdy'],Y_c)
            loss+=loss1
           
            loss_count += loss.data
            loss.backward()
            if cur_it % 50 == 49:
                
                writer.add_scalar('loss_paper', loss_paper.item(), cur_it+1)
                writer.add_scalar('loss', loss.item(), cur_it+1)
                
                writer.add_scalar('compute_bdy_loss', loss1.item(), cur_it+1) 
               
                full_path = train_loader.images_list[idx_list[0]]
                filenames = os.path.basename(full_path)  
                base_filename = os.path.splitext(filenames)[0] 
                base_filename = f"{base_filename}_{epoch}"  
                img_grey = rgb2grey(images.clone()[0].unsqueeze(0))

                # print(image.shape)
                a_f=torch.sigmoid(Y['final'])
                a_d=Y['bdy']
                y_f=torch.unsqueeze(a_f[0],0)
                y_d=torch.unsqueeze(a_d[0],0)
                y_c=torch.unsqueeze(Y_c[0],0)
               
                y_select=torch.unsqueeze(selected_gts[0],0)
                y_gts=torch.unsqueeze(gts[0],0)
                y_gts1=torch.unsqueeze(gts1[0],0)
                y_gts2=torch.unsqueeze(gts2[0],0)
                
                
                image = torch.cat((img_grey,y_select,y_gts,y_gts1,y_gts2,y_f,y_d,y_c), 0)
                
                writer.add_images('sal maps', image, cur_it + 1, dataformats='NCHW')
               
                # save_dir = os.path.join("/home/zzx/USOD/A2S-USOD-main/huatu", base_filename)
                # os.makedirs(save_dir, exist_ok=True)
                
                # 
                # tensor_dict = {
                #     'image': img_grey,
                #     'y_select': y_select,
                #     'y_gts': y_gts,
                #     'y_gts1': y_gts1,
                #     'y_gts2': y_gts2,
                #     'y_f': y_f,
                #     'y_d': y_d,
                #     'y_c': y_c
                # }
                
                # 
                # for kind, tensor in tensor_dict.items():
                #     
                #     np_tensor = tensor.squeeze().cpu().detach().numpy()
                #    
                #     np_tensor = (np_tensor * 255).astype(np.uint8)
                #     
                #     img = Image.fromarray(np_tensor)
                #    
                #     img.save(os.path.join(save_dir, f"{kind}.png"))

                
            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0
            
            Bar.suffix = '{:4}/{:4} | loss: {:1.5f}, time: {}.'.format(i, iter_per_epoch, round(float(loss_count / i), 5), round(time.time() - st, 3))
            bar.next()
            
            if epoch > 3:

                avg_predictions = pnc_module.calculate_average_prediction(image_ids)
                epsilon = pnc_module.calculate_epsilon(avg_predictions)
                
                
                corrected_gts = pnc_module.correct_labels(image_ids, gts, avg_predictions, epsilon)
                corrected_gts1 = pnc_module.correct_labels(image_ids, gts1, avg_predictions, epsilon)
                corrected_gts2 = pnc_module.correct_labels(image_ids, gts2, avg_predictions, epsilon)
                
                
                train_loader.gts[idx_list] = corrected_gts.cpu().numpy()
                train_loader.gts1[idx_list] = corrected_gts1.cpu().numpy()
                train_loader.gts2[idx_list] = corrected_gts2.cpu().numpy()

        bar.finish()
        if trset in ('DUTS-TR', 'MSB-TR'):
            test_model(model, test_sets, config, epoch)
            

if __name__ == "__main__":
    main()