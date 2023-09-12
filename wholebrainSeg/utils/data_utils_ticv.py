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
    AsChannelFirstd,
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
    RandRotated
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

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label", "label_ticv", "label_pfv"]),
            AddChanneld(keys=["image", "label", "label_ticv", "label_pfv"]),
            RandSpatialCropd(keys=["image", "label", "label_ticv", "label_pfv"], roi_size=[
                            cfig['roi_x'], cfig['roi_y'], cfig['roi_z']], random_size=False),
            RandFlipd(keys=["image", "label", "label_ticv", "label_pfv"], prob=cfig['aug_type']['flip'], spatial_axis=0),
            RandFlipd(keys=["image", "label", "label_ticv", "label_pfv"], prob=cfig['aug_type']['flip'], spatial_axis=1),
            RandFlipd(keys=["image", "label", "label_ticv", "label_pfv"], prob=cfig['aug_type']['flip'], spatial_axis=2),
            
            RandRotated(keys=["image", "label", "label_ticv", "label_pfv"], prob=cfig['aug_type']['rotate'], range_x=[-0.4, 0.4], mode=['bilinear', 'nearest', 'nearest','nearest']),
            RandRotated(keys=["image", "label", "label_ticv", "label_pfv"], prob=cfig['aug_type']['rotate'], range_y=[-0.4, 0.4], mode=['bilinear', 'nearest', 'nearest','nearest']),
            RandRotated(keys=["image", "label", "label_ticv", "label_pfv"], prob=cfig['aug_type']['rotate'], range_z=[-0.4, 0.4], mode=['bilinear', 'nearest', 'nearest','nearest']),

            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True, dtype=np.float32), # important
            RandScaleIntensityd(keys="image", factors=0.1, prob=cfig['aug_type']['scale_intensity']),
        
            RandShiftIntensityd(keys="image", offsets=0.1, prob=cfig['aug_type']['shif_intensity']),
            ToTensord(keys=["image", "label", "label_ticv", "label_pfv"]),
        ]
    )
    
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label", "label_ticv", "label_pfv"]),
            AddChanneld(keys=["image", "label", "label_ticv", "label_pfv"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True, dtype=np.float32),
            ToTensord(keys=["image", "label","label_ticv","label_pfv"]),
        ])
    
    data_dir = cfig['data_dir']
    jsonlist = os.path.join(cfig['jsondir'], 'fold{}.json'.format(cfig['fold']))
    datalist = load_decathlon_datalist(
        jsonlist, True, "training", base_dir=data_dir)
    val_files = load_decathlon_datalist(
        jsonlist, True, "validation", base_dir=data_dir)

    train_ds = SmartCacheDataset(
        data=datalist, transform=train_transforms, replace_rate=1.0, cache_num=30)

    train_loader = DataLoader(
        train_ds, batch_size=cfig['batch_size'], shuffle=True)
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    val_shape_dict = {}

    for d in val_files:
        imagepath = d["image"]
        imagename = imagepath.split('/')[-1]
        imgnb = nb.load(imagepath)
        val_shape_dict[imagename] = [
            imgnb.shape[0], imgnb.shape[1], imgnb.shape[2]]
    print('Totoal number of validation: {}'.format(len(val_shape_dict)))

    return train_loader, val_loader, val_shape_dict
