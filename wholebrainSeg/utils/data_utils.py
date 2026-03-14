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
import nibabel as nb
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    AddChanneld,
    Compose,
    MapTransform,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd,
    NormalizeIntensityd,
    RandAffined,
    RandAdjustContrastd,
    CenterSpatialCrop,
    CenterSpatialCropd,
    ScaleIntensityRangePercentilesd,
    CropForeground, 
    RandRotated,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandZoomd
)

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    load_decathlon_properties,
    partition_dataset,
    select_cross_validation_folds,
    SmartCacheDataset,
    Dataset,
    decollate_batch,
)
from monai.data import CacheDataset, SmartCacheDataset, DataLoader, Dataset
import numpy as np

nb.Nifti1Header.quaternion_threshold = -1e-06


def get_loader(cfig):

    # 获取增强概率配置
    aug = cfig.get('aug_type', {})
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            RandSpatialCropd(keys=["image", "label"], roi_size=[
                            cfig['roi_x'], cfig['roi_y'], cfig['roi_z']], random_size=False),
            # 翻转增强
            RandFlipd(keys=["image", "label"], prob=aug.get('flip', 0.5), spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=aug.get('flip', 0.5), spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=aug.get('flip', 0.5), spatial_axis=2),
            
            # 旋转增强（增加旋转角度范围）
            RandRotated(keys=["image", "label"], prob=aug.get('rotate', 0.3), range_x=[-0.5, 0.5], mode=['bilinear', 'nearest']),
            RandRotated(keys=["image", "label"], prob=aug.get('rotate', 0.3), range_y=[-0.5, 0.5], mode=['bilinear', 'nearest']),
            RandRotated(keys=["image", "label"], prob=aug.get('rotate', 0.3), range_z=[-0.5, 0.5], mode=['bilinear', 'nearest']),
            
            # 缩放增强
            RandZoomd(keys=["image", "label"], prob=aug.get('zoom', 0.2), 
                     min_zoom=0.9, max_zoom=1.1, mode=['trilinear', 'nearest'], keep_size=True),

            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True, dtype=np.float32),
            
            # 强度增强
            RandScaleIntensityd(keys="image", factors=0.15, prob=aug.get('scale_intensity', 0.3)),
            RandShiftIntensityd(keys="image", offsets=0.15, prob=aug.get('shift_intensity', 0.3)),
            
            # 新增增强：高斯噪声
            RandGaussianNoised(keys="image", prob=aug.get('gaussian_noise', 0.2), std=0.05),
            
            # 新增增强：高斯平滑
            RandGaussianSmoothd(keys="image", prob=aug.get('gaussian_smooth', 0.2), 
                              sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            
            # 新增增强：对比度调整
            RandAdjustContrastd(keys="image", prob=aug.get('contrast', 0.3), gamma=(0.8, 1.2)),
            
            ToTensord(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=False),
            EnsureChannelFirstd(keys=["image", "label"], channel_dim='no_channel'),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True, dtype=np.float32),
            ToTensord(keys=["image", "label"]),
        ]
        )
    data_dir = cfig['data_dir']
    jsonlist = os.path.join(cfig['jsondir'], 'fold{}.json'.format(cfig['fold']))

    datalist = load_decathlon_datalist(
        jsonlist, True, "training", base_dir=data_dir)
    val_files = load_decathlon_datalist(
        jsonlist, True, "validation", base_dir=data_dir)

    # 从配置读取参数，采用最保守的稳定配置
    cache_rate = cfig.get('cache_rate', 0.0)  # 默认禁用缓存以减少内存占用
    num_workers = cfig.get('num_workers', 0)  # 默认单进程
    
    train_ds = CacheDataset(
        data=datalist, 
        transform=train_transforms, 
        cache_rate=cache_rate, 
        num_workers=num_workers
    )
    
    # DataLoader配置：最保守配置确保稳定性
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfig['batch_size'], 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=False,  # 禁用pin_memory减少内存压力
        persistent_workers=False
    )

    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=0.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    val_shape_dict = {}

    for d in val_files:
        imagepath = d["image"]
        imagename = imagepath.split('/')[-1]
        imgnb = nb.load(imagepath)
        val_shape_dict[imagename] = [
            imgnb.shape[0], imgnb.shape[1], imgnb.shape[2]]
    print('Totoal number of validation: {}'.format(len(val_shape_dict)))

    return train_loader, val_loader, val_shape_dict
