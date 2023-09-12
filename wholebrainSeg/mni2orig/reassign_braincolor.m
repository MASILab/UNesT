function reassign_braincolor(almost_final_Seg,final_Seg)

if ~exist(final_Seg)
    
    final_dir = fileparts(final_Seg);
    if ~isdir(final_dir);mkdir(final_dir);end;
    
    labels = [4,11,23,30,31,32,35,36,37,38,39,40,41,44,45,47,48,49,50,51,52,55,56,57,58,59,60,61,62,71,72,73,75,76,100,101,102,103,104,105,106,107,108,109,112,113,114,115,116,117,118,119,120,121,122,123,124,125,128,129,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207];
    
    almost_nii = load_untouch_nii_gz(almost_final_Seg);
    almost_img = almost_nii.img;
    final_img = ones(size(almost_img))*255;
    final_img(almost_img==0) = 0;
    for li = 1:length(labels)
        final_img(almost_img==li) = labels(li);
    end
    
    final_nii = almost_nii;
    final_nii.img = final_img;
    save_untouch_nii_gz(final_nii,final_Seg);
end

end