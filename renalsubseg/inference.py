"""Inference script for UNesT.

Inference the images in the tested set and saved the output in .npy format for emsemble.
"""
import os
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai import transforms, data
import nibabel as nib
import scipy.ndimage as ndimage
import argparse
from skimage.measure import label
from networks.unest_base_patch_4 import UNesT
from monai.config import DtypeLike, KeysCollection
nib.Nifti1Header.quaternion_threshold = -1e-06


parser = argparse.ArgumentParser(description='UNesT Testing')
parser.add_argument('--imagesTs_path', type=str, help='path to the testing images')
parser.add_argument('--saved_model_path', type=str)
parser.add_argument('--base_dir', type=str, help='Path to save the results')
parser.add_argument('--fold', default=0, type=int)
parser.add_argument('--sw_batch_size', default=1, type=int)
parser.add_argument('--overlap', default=0.7, type=float, help='Overlap for the sliding window inference.')
parser.add_argument('--device', default=0, type=int)
args = parser.parse_args()
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
### -----------------------------------------------------------------

# set the output saving folder
base_save_pred_dir = os.path.join(args.base_dir, 'pred_{}/'.format(args.overlap))

model_name = 'fold{}'.format(args.fold)
checkpoint_dir = args.saved_model_path

path = args.imagesTs_path

###-------------------------------------------------------------------
results_folder = base_save_pred_dir + model_name
checkpoints = [checkpoint_dir]

if not os.path.exists(base_save_pred_dir):
    os.makedirs(base_save_pred_dir)

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
# ----------------------------------------------------------------------------

# set testing list with specific name. 

ids = []
validation_files = []
files = os.listdir(path)
for file in files:
    if not file.startswith('.'):
        img_id = file.split('_')[0].split('.nii.gz')[0]
        if img_id not in ids:
            ids.append(img_id)
            validation_files.append({'label': '',
            'image': [os.path.join(path, file)]})

val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.AddChanneld(keys=["image"]),
        transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        transforms.SpatialPadd(keys=["image"], mode=["minimum"], spatial_size=[96, 96, 96]),
        transforms.ToTensord(keys=["image"]),
    ]
)

# import and load models
img_size = (96,96,96)


model = UNesT(in_channels=1,
            out_channels=4,
            patch_size=4,
            depths=[2, 2, 8],
            num_heads=[4, 8, 16],
            embed_dim=[128, 256, 512]
        )

ckpt = torch.load(checkpoint_dir, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=True)
model.to(device)

model.eval()

val_ds = data.Dataset(data=validation_files, transform=val_transforms)
val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, sampler=None)
overlap_ratio = args.overlap
sw_batch_size = args.sw_batch_size

# run testing iteratively
with torch.no_grad():
    i = 0
    for idx, batch_data in enumerate(val_loader):
        case_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii.gz')[0]

        print('##############  Inference case {}-{}  ##############'.format(idx, case_name))
        image = batch_data['image'].to(device)

        affine = batch_data['image_meta_dict']['original_affine'][0].numpy()
        infer_outputs = 0.0

        pred = sliding_window_inference(image, img_size, sw_batch_size, model, overlap=overlap_ratio, device=torch.device('cpu'),mode='gaussian')
        infer_outputs += torch.nn.Softmax(dim=1)(pred) # you may need to use softmax according to each task
        infer_outputs = infer_outputs.cpu().numpy()
        case_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii.gz')[0]
        subject_folder = os.path.join(results_folder, case_name)
        outNUMPY = results_folder + '/' + case_name + '.npy'
        np.save(outNUMPY, infer_outputs)
