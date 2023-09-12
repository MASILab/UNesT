# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from turtle import circle
import nibabel as nb
from PIL import Image
import scipy.ndimage as ndimage
import numpy as np
import torch
import torch.nn as nn
import argparse
from tensorboardX import SummaryWriter
from monai.losses import DiceLoss,DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete,Activations,Compose
from tqdm import tqdm
from utils.data_utils import get_loader
from optimizers.lr_scheduler import WarmupCosineSchedule

import pdb
import yaml

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def Dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

def resample(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = ( float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom( img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

def main(cfig, device):
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(global_step,train_loader,dice_val_best, val_shape_dict):
        model.train()
        epoch_iterator = tqdm(train_loader,desc="Training (X / X Steps) (loss=X.X)",dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            x, y = (batch["image"].to(device), batch["label"].to(device))
            logit_map = model(x)

            # training Dice 
            if global_step % 40 == 0:
                train_pred = torch.softmax(logit_map, 1).detach().cpu().numpy()
                train_pred = np.argmax(train_pred, axis = 1).astype(np.uint8)
                train_label = y.detach().cpu().numpy()[:,0,:,:,:]
                
                dice_list_sub = []
                for i in range(1, cfig['num_classes']):
                    organ_Dice = Dice(train_pred[0] == i, train_label[0] == i)
                    dice_list_sub.append(organ_Dice)
                print('Train DSC: {}'.format(np.mean(dice_list_sub)))
                writer.add_scalar("train/DSC_sample", scalar_value=np.mean(dice_list_sub), global_step=global_step)

            #----------------------------------------
            try:
                loss = loss_function(logit_map, y)
            except:

                loss = loss_function(logit_map, y[:,0].long())
            
            loss.backward()
            optimizer.step()
            if cfig['lrdecay']:
                scheduler.step()
            optimizer.zero_grad()
            epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, cfig['num_steps'], loss))
            writer.add_scalar("train/loss", scalar_value=loss, global_step=global_step)

            global_step += 1
            if global_step % cfig['eval_num'] == 0:
                epoch_iterator_val = tqdm(test_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)

                mean_list = validation(epoch_iterator_val, val_shape_dict)

                writer.add_scalar("ValAvgDice/Dice_avg", scalar_value=np.mean(mean_list), global_step=global_step)
                for lbl_i in range(cfig['num_classes']-1):
                    writer.add_scalar("Validation/Dice_{}".format(lbl_i+1), scalar_value=mean_list[lbl_i], global_step=global_step)


                dice_val = np.mean(mean_list)
                if dice_val > dice_val_best:
                    checkpoint = {'global_step': global_step, 'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict()}
                    save_ckp(checkpoint, logdir + '/model.pt')
                    dice_val_best = dice_val
                    print('Model Was Saved ! Current Best Dice: {},  Current Dice: {}'.format(dice_val_best, np.mean(mean_list)))
                else:
                    print('Model Was NOT Saved ! Current Best Dice: {} Current Dice: {}'.format(dice_val_best, dice_val))
        return global_step, dice_val_best


    def validation(epoch_iterator_val, val_shape_dict):
        model.eval()
        metric_values = []
        roi_size = (cfig['roi_x'], cfig['roi_y'], cfig['roi_z'])
        sw_batch_size = cfig['sw_batch_size']
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
                name = batch["image_meta_dict"]['filename_or_obj'][0].split('/')[-1]

                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, overlap=0.2, device=torch.device('cpu'))
                val_outputs = torch.softmax(val_outputs, 1).detach().cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis = 1).astype(np.uint8)
                val_labels = val_labels.detach().cpu().numpy()[:,0,:,:,:]

                dice_list_sub = []
                for i in range(1, cfig['num_classes']):
                    organ_Dice = Dice(val_outputs == i, val_labels == i)
                    dice_list_sub.append(organ_Dice)

                dice_mean = np.mean(dice_list_sub)
                metric_values.append(dice_list_sub)
                epoch_iterator_val.set_description("Validate (%d / %d Steps) (dice_mean=%2.5f)" % (global_step, 5.0, dice_mean))
                # -------------------------------------------------

            mean_list = np.mean(metric_values, axis=0)

        return mean_list

    
    torch.backends.cudnn.benchmark = True
    cfig['n_gpu'] = torch.cuda.device_count()
    cfig['device'] = device

    if cfig['model_type'] == 'base':
        from networks.unest_base_patch_4 import UNesT
    elif cfig['model_type'] == 'small':
        from networks.unest_small_patch_4 import UNesT
    elif cfig['model_type'] == 'large':
        from networks.unest_large_patch_4 import UNesT
    model = UNesT(in_channels=1,
                        out_channels=14,
                        patch_size=cfig['patch_size'],
                        depths=cfig['depth'],
                        num_heads=cfig['num_heads'],
                        embed_dim=cfig['embed_dims']
                    ).to(device)
    model.to(device)

    num_params = count_parameters(model)
    print("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)

    logdir = cfig['logdir']
    writer = SummaryWriter(logdir=logdir)

    if cfig['opt'] == "adam":
        optimizer = torch.optim.Adam(params = model.parameters(), lr=cfig['lr'],weight_decay= cfig['decay'])

    elif cfig['opt'] == "adamw":
        optimizer = torch.optim.AdamW(params = model.parameters(), lr=cfig['lr'], weight_decay=cfig['decay'])

    elif cfig['opt'] == "sgd":
        optimizer = torch.optim.SGD(params = model.parameters(), lr=cfig['lr'], momentum=cfig['momentum'], weight_decay=cfig['decay'])

    if cfig['lrdecay']:
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=cfig['warmup_steps'], t_total=cfig['num_steps'])

    if cfig['loss_type'] == 'dice_ce':
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=False, smooth_nr=0, smooth_dr=1e-6)
    elif cfig['loss_type'] == 'dice':
        loss_function = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-6)
    elif cfig['loss_type'] == 'ce':
        loss_function = nn.CrossEntropyLoss()
    elif cfig['loss_type'] == 'wce':
        weight = np.ones(133).tolist()
        for w in cfig['weight_classes']:
            weight[w] = 10.0

        class_weights = torch.FloatTensor(weight).to(device)
        loss_function = nn.CrossEntropyLoss(weight=class_weights)
    elif cfig['loss_type'] == 'dice_wce':
        weight = np.ones(133).tolist()
        for w in cfig['weight_classes']:
            weight[w] = 10.0
        class_weights = torch.FloatTensor(weight).to(device)
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=False, smooth_nr=0, smooth_dr=1e-6, ce_weight=class_weights)

    train_loader, test_loader, val_shape_dict = get_loader(cfig)
    global_step = 0
    dice_val_best = 0.0

    while global_step < cfig['num_steps']:
        global_step, dice_val_best = train(global_step,train_loader,dice_val_best, val_shape_dict)
    checkpoint = {'global_step': global_step,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
    save_ckp(checkpoint, logdir+'/model_final_epoch.pt')

if __name__ == '__main__':

    yaml_file = 'multiorganseg/yaml/mulorganseg_large.yaml'
    with open(yaml_file, 'r') as f:
        cfig = yaml.safe_load(f)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main(cfig, device)
