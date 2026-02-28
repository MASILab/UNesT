# 文件内容介绍：
/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/UIH164下为原始影像和标注结果，包含多个患者，如76384925062202为患者编号；

/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/UIH164/76384925062202/dcm目录下为原始DICOM影像；

/home/tenoke4090/B_WorkPath/mrqs/wholebrainseg_dataset/UIH164/76384925062202/c_results目录下，brain_preproc_img.nii.gz 为原始影像转到mini空间并区颅骨的影像；cleanup_labelmap96_src.nii.gz， 为brain_preproc_img.nii.gz对应的96空间下的标签结果


处理步骤：
1. 获取原始DICOM影像，并转成nii.gz格式
2. 获取原始dicom影像空间和brain_preproc_img.nii.gz空间之间的映射关系
3. 根据映射关系，将cleanup_labelmap96_src.nii.gz映射到原始影像空间