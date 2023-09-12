# --------------------------------------------------------
# Copyright (C) 2021 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Transformer Pretraining Code: Yucheng, Vishwesh, Ali
# --------------------------------------------------------

import os
import numpy as np
from numpy.random import randint
from PIL import Image
import nibabel as nb
import json

# Generate JSON file


import os

traindir = './train/images'
valdir = './validation/images'
json_file = './renalseg.json'

sublist = [s for s in os.listdir(traindir)]
allnum = len(sublist)

datadict = {}
datadict['training'] = []
datadict['validation'] = []

for f in sublist:
    
    ifile = "train/images/" + f
    t_dict = {"image": '', 'label':''}

    t_dict['image'] = ifile
    f_segname = f
    
    ilabel = "train/labels/" + f_segname 
    t_dict['label'] = ilabel

    datadict['training'].append(t_dict)
    

sublist = [s for s in os.listdir(valdir)]
allnum = len(sublist)

for f in sublist:

    ifile = "validation/images/" + f
    t_dict = {"image": '', 'label':''}

    t_dict['image'] = ifile
    f_segname = f

    ilabel = "validation/labels/" + f_segname 
    t_dict['label'] = ilabel

    datadict['validation'].append(t_dict)


with open(json_file, 'w') as f:
    json.dump(datadict, f, indent=4, sort_keys=True)
