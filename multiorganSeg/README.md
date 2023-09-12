# Multi-organ segmentation (BTCV)

## Prepare Training Data
---
We convert the training data list into json file using [create_json.py](utils/create_json.py).

The training and validation data are saved as follow:

    train/
        ├── images
        ├── labels
    validation/    
        ├── images
        ├── labels

## Training
---
We specifiy the model hyperparmeters in the yaml files, sample yaml files for different model scales can be found in [yaml](yaml) folder. 

    yaml/
        ├── unest_base
        ├── unest_large
        ├── unest_small

Before training, specify the paths in the yaml files and set the yaml path in the [main.py](main.py). 

## Inference
---
We use [inference.py](inference.py) to test each images in the testing set and save the final output into Nifti format. The trained model weight can found here: [multiorganseg_weight](https://drive.google.com/file/d/175PGZUbaEefIdZ3oiCIBkxTxM7O2kv5a/view?usp=sharing).

Running inference:
```
python inference.py --imagesTs_path test_images_path --saved_model_path path2saved_model --base_dir output_path --fold 0 --overlap 0.7 --device 0
```

## Citation
---
If you find this repository useful, please consider citing the following papers:

```
@article{yu2023unest,
  title={UNesT: local spatial representation learning with hierarchical transformer for efficient medical segmentation},
  author={Yu, Xin and Yang, Qi and Zhou, Yinchi and Cai, Leon Y and Gao, Riqiang and Lee, Ho Hin and Li, Thomas and Bao, Shunxing and Xu, Zhoubing and Lasko, Thomas A and others},
  journal={Medical Image Analysis},
  pages={102939},
  year={2023},
  publisher={Elsevier}
}
```