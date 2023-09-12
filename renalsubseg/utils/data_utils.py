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
import nibabel as nb
import os
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
from monai.data import CacheDataset,SmartCacheDataset, DataLoader, Dataset
import numpy as np


def get_loader(cfig):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], 
                        spatial_size=[cfig['roi_x'], cfig['roi_y'], cfig['roi_z']]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(cfig['roi_x'], cfig['roi_y'], cfig['roi_z']),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.15, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.15, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.15, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], 
                        spatial_size=(cfig['roi_x'], cfig['roi_y'], cfig['roi_z'])),
            ToTensord(keys=["image", "label"]),
        ]
    )


    data_dir = cfig['datadir'] 
    jsonlist = cfig['jsonfile']


    datalist = load_decathlon_datalist(jsonlist, True, "training", base_dir=data_dir)
    val_files = load_decathlon_datalist(jsonlist, True, "validation", base_dir=data_dir)

    train_ds = SmartCacheDataset(data=datalist, transform=train_transforms, replace_rate=1.0, cache_num=20)

    train_loader = DataLoader(train_ds, batch_size=cfig['batch_size'], shuffle=True)
    val_ds = Dataset(data=val_files, transform=val_transforms)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    val_shape_dict = {}
    
    for d in val_files:
        imagepath = d["image"]
        imagename = imagepath.split('/')[-1]
        imgnb = nb.load(imagepath)
        val_shape_dict[imagename] = [imgnb.shape[0], imgnb.shape[1], imgnb.shape[2]]
    print('Totoal number of validation: {}'.format(len(val_shape_dict)))
    
    return train_loader, val_loader, val_shape_dict
