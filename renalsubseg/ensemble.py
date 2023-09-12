""" Ensemble results from different folds and save them into Nifti format."""


import os
import numpy as np
from monai import transforms, data
import nibabel as nib
import argparse
import scipy.ndimage as ndimage
from skimage.measure import label

#------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='UNesT Ensemble')
parser.add_argument('--prob_dir', default='./pred_0.7', type=str, help='path to the folder contains probability of different folds')
parser.add_argument('--img_path', default='./images', type=str, help='Test image path')
parser.add_argument('--out_path', default='./results', type=str, help='Path to save the outputs')
args = parser.parse_args()
#---------------------------------------------------------------------------------------------

def resize_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = ( float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom( img, zoom_ratio, order=0, prefilter=False)
    return img_resampled
def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


base_save_pred_dir = args.prob_dir
path = args.img_path
results_folder = args.out_path
model_names = os.listdir(base_save_pred_dir)


if not os.path.exists(results_folder):
    os.makedirs(results_folder)

subject_list = os.listdir(os.path.join(base_save_pred_dir, model_names[0]))

count = 0
for sub in subject_list:
    ct_model = len(model_names)

    infer_outputs = 0.0
    for model_name in model_names:
        pred_file = os.path.join(base_save_pred_dir, model_name, sub)
        pred_numpy = np.load(pred_file)
        infer_outputs += pred_numpy

    infer_outputs = infer_outputs / ct_model
    probs = infer_outputs

    labels = np.argmax(probs, axis=1).astype(np.uint8)[0]  # get discrete lables for each class

    sub = sub.split('.npy')[0]
    case_name = sub+'.nii.gz'
    output_file = case_name

    original_file = nib.load(os.path.join(path, output_file))
    original_affine = original_file.affine

    target_shape = original_file.shape
    print('target shape: {}'.format(target_shape))

    labels = resize_3d(labels, target_shape)

    count += 1
    print('[{}] label finished {}'.format(count, sub))
    nib.save(nib.Nifti1Image(labels.astype(np.uint8), original_affine), os.path.join(results_folder, output_file))
