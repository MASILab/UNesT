% run on docker 
function output = Run_Deep_brain_postprocessing(mdir,final_out_dir,result_dir,orig_img_dir,mni_img_dir,ants_dir,niftyreg_dir)
    target_dir = orig_img_dir;
    img_dir = mni_img_dir;
    extra_dir = '/extra';

    in.ants_loc = ants_dir;
    in.niftyreg_loc = niftyreg_dir;
    
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
            orig_inv_seg_file = postproc_pipeline(target_fname,working_dir,img_dir,result_dir,in);
            system(sprintf('cp %s %s',orig_inv_seg_file,output_final));
        end
    end
end