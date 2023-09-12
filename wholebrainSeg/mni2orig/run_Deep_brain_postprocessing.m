clc;clear;close all;

% addpath(genpath('/fs4/masi/huoy1/FS3_backup/software/full-multi-atlas/masi-fusion/src'));
% addpath(genpath('/share4/huoy1/Deep_5000_Brain/code/evaluation'));

% test on local machine
% mdir = '/share4/xiongy2/docker/OUTPUTS';
% final_out_dir = '/share4/xiongy2/docker/OUTPUTS/FinalResult'; 
% target_dir = '/share4/xiongy2/docker/INPUTS';
% extra_dir = '/share4/xiongy2/docker/extra';

% run on docker 
mdir = '/nfs/masi/yux11/UNesT/wholeBrainSeg2/inference/UNesT/5fold/mni_new_pretrain/pred_orig_0.9_clip_bg_0.8_fromhuo';
final_out_dir = '/nfs/masi/yux11/UNesT/wholeBrainSeg2/inference/UNesT/5fold/mni_new_pretrain/final_resutls_orig_0.9_clip_bg_0.8_fromhuo'; 
target_dir = '/nfs/masi/yux11/UNesT/wholeBrainSeg2/data/ori_45/processed/test/images';
img_dir = '/nfs/masi/yux11/UNesT/wholeBrainSeg2/data/mni_45/test/images';
result_dir = '/nfs/masi/yux11/UNesT/wholeBrainSeg2/inference/UNesT/5fold/mni_new_pretrain/results_mni_0.9_clip_bg_0.8';
extra_dir = '/extra';

% in.ants_loc = [extra_dir filesep 'full-multi-atlas' filesep 'ANTs-bin' filesep];
in.ants_loc = '/nfs/masi/caily/apps/ants/';
in.niftyreg_loc = '/usr/local/bin';
% in.niftyreg_loc =  [extra_dir filesep 'full-multi-atlas' filesep 'niftyreg' filesep 'bin' filesep];

sublist = dir([target_dir filesep '*.nii.gz']);

if ~isdir(final_out_dir);mkdir(final_out_dir);end;

for si = 1:length(sublist)
    subFile = sublist(si).name;
    subName = get_basename(subFile);
    target_fname = [target_dir filesep subFile];
    % tic;
    normed_dir = [mdir filesep subName];
    working_dir = [normed_dir filesep 'working_dir'];
    normed_file = [normed_dir filesep 'target_processed.nii.gz'];
   
    output_final = [final_out_dir filesep sprintf('%s_seg.nii.gz',subName)];
    if ~exist(output_final)
%         print('===')
        orig_inv_seg_file = postproc_pipeline(target_fname,working_dir,img_dir,result_dir,in);
        system(sprintf('cp %s %s',orig_inv_seg_file,output_final));
    end
end