"""Inference script for UNesT with TICV/PFV estimation.

Inference the images in the tested set and saved the output in Nifti format.
"""
import os
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai import transforms, data
import nibabel as nib
import scipy.ndimage as ndimage
import argparse
from scipy.ndimage.measurements import label
from networks.unest import UNesT_ticv
nib.Nifti1Header.quaternion_threshold = -1e-06


parser = argparse.ArgumentParser(description='UNesT with TICV/PFV Testing')
parser.add_argument('--imagesTs_path', type=str, help='path to the testing images')
parser.add_argument('--saved_model_path', type=str)
parser.add_argument('--fold', default=0, type=int)
parser.add_argument('--sw_batch_size', default=1, type=int)
parser.add_argument('--overlap', default=0.7, type=float, help='Overlap for the sliding window inference.')
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--results_folder_brain', type=str)
parser.add_argument('--results_folder_ticv', type=str)
parser.add_argument('--results_folder_pfv', type=str)

args = parser.parse_args()
torch.backends.cudnn.benchmark = True

if not os.path.exists(args.results_folder_brain):
    os.makedirs(args.results_folder_brain)
if not os.path.exists(args.results_folder_ticv):
    os.makedirs(args.results_folder_ticv)
if not os.path.exists(args.results_folder_pfv):
    os.makedirs(args.results_folder_pfv)

device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
### -----------------------------------------------------------------

# set the saved model path
checkpoint_dir = args.saved_model_path

# set the test image path 
path = args.imagesTs_path

# set testing list with specific name. 

ids = []
validation_files = []
files = os.listdir(path)
for file in files:

    validation_files.append({'label': '',
    'image': [os.path.join(path, file)]})

val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.AddChanneld(keys=["image"]),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.ToTensord(keys=["image"]),
    ]
)

# import and load models
img_size = (96,96,96)
# need to set the correct model class

model = UNesT_ticv(in_channels=1,
                out_channels=133,
                )

ckpt = torch.load(checkpoint_dir, map_location='cpu')
state_dict = {k: v for k, v in ckpt['state_dict'].items() if not ('total_ops' in k or 'total_params' in k)}
model.load_state_dict(state_dict, strict=True)
model.to(device)
model.named_parameters()

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

        pred = sliding_window_inference(image, img_size, sw_batch_size, model, overlap=overlap_ratio, device=torch.device('cpu'))
        infer_outputs += torch.nn.Softmax(dim=1)(pred[:,:133]) # you may need to use softmax according to each task
        infer_outputs = infer_outputs.cpu().numpy()
        case_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii.gz')[0]

        infer_outputs_ticv = torch.nn.Sigmoid()(pred[:,133]).cpu().numpy()
        infer_outputs_ticv = np.array(infer_outputs_ticv > 0.5).astype(np.int)[0]

        struct = ndimage.generate_binary_structure(3, 3)
        label_ticv = np.zeros(infer_outputs_ticv.shape, dtype=np.int)
        
        labeled_prediction, n_components = label(infer_outputs_ticv, struct)
        sizes = ndimage.sum(infer_outputs_ticv, labeled_prediction, range(1, n_components + 1))
        max_label = np.where(sizes == sizes.max())[0] + 1
        label_ticv[labeled_prediction==max_label] = 1
        label_ticv = label_ticv.astype(np.float32)

        infer_outputs_pfv = torch.nn.Sigmoid()(pred[:,134]).cpu().numpy()
        infer_outputs_pfv = np.array(infer_outputs_pfv > 0.5).astype(np.uint8)[0]
        
        labels = np.argmax(infer_outputs, axis=1).astype(np.uint8)[0]
        original_file = nib.load(os.path.join(path, case_name + '.nii.gz'))
        original_affine = original_file.affine

        # saved to Nifti format
        nib.save(nib.Nifti1Image(labels.astype(np.uint8), original_affine), os.path.join(args.results_folder_brain, case_name + '.nii.gz'))
        nib.save(nib.Nifti1Image(label_ticv.astype(np.uint8), original_affine), os.path.join(args.results_folder_ticv, case_name + '.nii.gz'))
        nib.save(nib.Nifti1Image(infer_outputs_pfv.astype(np.uint8), original_affine), os.path.join(args.results_folder_pfv, case_name + '.nii.gz'))

