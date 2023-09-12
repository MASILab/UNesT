#!/bin/bash

#output
mdir="./pred_orig" # working directory
final_out_dir="./final_resutls" # final predictions in the original space

#input
result_dir="/results_mni" # Nifty file for the prediction in MNI space
orig_img_dir="./orig/images" # Path to the original images
mni_img_dir="./mni/images" # Path to the images in the MNI space

ants_dir="" # path to the ANTs
niftyreg_dir="" #path to the NiftyReg

cd ./wholebrainSeg/mni2orig
matlab -nodisplay -r "Run_Deep_brain_postprocessing('${mdir}','${final_out_dir}','${result_dir}','${orig_img_dir}','${mni_img_dir}','${ants_dir}','${niftyreg_dir}'),quit()"
