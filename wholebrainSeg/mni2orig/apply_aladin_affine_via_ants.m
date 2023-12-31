function [varargout] = apply_aladin_affine_via_ants(NR_bin_dir, ...
                                                    ANTs_bin_dir, ...
                                                    out_fname, ...
                                                    in_raw_fname, ...
                                                    in_seg_fname, ...
                                                    target_fname, ...
                                                    out_dir, tmp_dir)
% APPLY_ALADIN_AFFINE_VIA_ANTS - applys a reg_aladin affine transformation via
%                                the better ANTs interpolator
%
% apply_aladin_affine_via_ants(ANTS_bin_dir, out_fname, aladin_txt_fname,...
%                              in_raw_fname, in_seg_fname, target_fname)
%
% Input: NR_bin_dir - the location of the nifty reg binaries
%        ANTs_bin_dir - the location of the ANTS binaries
%        out_fname - the output segmentation filename
%        in_raw_fname - the image of the labels to be transformed
%        in_seg_fname - the laels to be transformed
%        target_fname - the target to transform the labels to
%        out_dir - the output directory
%        tmp_dir - the temporary directory
%
%

if ~exist(out_dir, 'dir'), mkdir(out_dir); end
if ~exist(tmp_dir, 'dir'), mkdir(tmp_dir); end

out_bname = [out_dir, '/inverse-MNI-reg'];
out_aff_mat = [out_bname, '0GenericAffine.mat'];
out_tmp_MNI_fname = [out_dir, '/tmp-target.nii.gz'];
out_tmp_MNI_SEG_fname = [out_dir, '/tmp-target-seg.nii.gz'];
out_tmp_aff_fname = [out_dir, '/tmp-affine.txt'];
txt_out = [tmp_dir, '/inverse-MNI-reg-output.txt'];

% print some stuff to the screen
if nargout == 0
    tprintf('-> Computing ANTs transformation back to original space\n');
    tprintf('Output Labels: %s\n', out_fname);
    tprintf('Output Affine Matrix: %s\n', out_aff_mat);
    tprintf('Output Text File: %s\n', txt_out);
end

% ANTs parameters
ANTs_initial_parms = sprintf('-u -r [%s,%s,1]', ...
                             out_tmp_MNI_SEG_fname, in_seg_fname);

% -- stage 1
ANTs_metric_parms1 = sprintf(['-m MeanSquares[%s,%s,1,3,Regular,0.5] ', ...
                              '-t Rigid[0.5]'], ...
                             out_tmp_MNI_SEG_fname, in_seg_fname);
ANTs_convergence_parms1 = '-c 1000x1000x1000 -f 6x4x2 -s 8x8x8';

% -- stage 2
ANTs_metric_parms2 = sprintf(['-m MeanSquares[%s,%s,1,3,Regular,0.25] ', ...
                              '-t Affine[0.5]'], ...
                             out_tmp_MNI_SEG_fname, in_seg_fname);
ANTs_convergence_parms2 = '-c 1000x1000x10 -f 4x2x1 -s 2x1x0';

% create the commands
cmds{1} = sprintf('%s/reg_aladin -ref %s -flo %s -res %s -aff %s > %s\n', ...
                  NR_bin_dir, target_fname, in_raw_fname, ...
                  out_tmp_MNI_fname, out_tmp_aff_fname, txt_out);
cmds{2} = sprintf(['%s/reg_resample -ref %s -flo %s -aff %s ', ...
                   '-res %s -inter 0 >> %s\n'], ...
                   NR_bin_dir, target_fname, in_seg_fname, ...
                   out_tmp_aff_fname, out_tmp_MNI_SEG_fname, txt_out);
cmds{3} = sprintf(['. ~/.bashrc && %s/antsRegistration -d 3 -o %s %s ', ...
                  '%s %s ', ...
                  '%s %s >> %s'], ...
                  ANTs_bin_dir, out_bname, ANTs_initial_parms, ...
                  ANTs_metric_parms1, ANTs_convergence_parms1, ...
                  ANTs_metric_parms2, ANTs_convergence_parms2, ...
                  txt_out);
cmds{4} = sprintf(['. ~/.bashrc && %s/antsApplyTransforms -d 3 -i %s ', ...
                   '-o %s -r %s -n MultiLabel -t %s >> %s\n'], ...
                  ANTs_bin_dir, in_seg_fname, out_fname, ...
                  target_fname, out_aff_mat, txt_out);

if nargout == 0
    run_cmd_single(cmds);
elseif nargout == 1
    varargout{1} = cmds;
else
    error('too many output arguments');
end
