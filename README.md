# vehicle_reid_itsc2023
## Strength in Diversity: Multi-Branch Representation Learning for Vehicle Re-Identification

Version 0.2 - Still requiring some cleaning and LAI

All models are trained with CUDA 11.3 and PyTorch 1.11 on RTX4090 and Ryzen 7950X.

Resuls are displayed as mAP / CMC1 in percentage values %.
VehicleID was not available at the time, we report values for VehicleID now. We follow evaluation as FastReid with 10-fold cross-validation to select queries and gallery.

To train:
```console
python main.py --model_arch MBR_4G --config ./config/config_VERIWILD.yaml
```

Test:
```console
python teste.py --path_weights ./logs/Veri776/MBR_4G/0/
```



### Veri-776
Full Precision - Baseline mAP: 81.15 CMC1: 96.96 lambda:0.6 beta:1.0

| R50      | 4G          | 2G          | 2X2G        | 2X          | 4X          |
| -------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| CE + Tri | 82.81/97.38 | 83.04/97.14 | 83.67/97.32 | 81.82/96.96 | 82.31/97.32 |
| CE / Tri | 82.47/96.84 | 83.26/97.02 | 84.22/97.02 | 83.67/97.50 | 83.89/97.5  |

| R50+BoT  | 4G          | 2G           | 2X2G        | 2X          | 4X          |
| -------- | ----------- | ------------ | ----------- | ----------- | ----------- |
| CE + Tri | 82.04/96.96 | 81.14/ 97.02 | 82.02/96.78 | 82.82/97.20 | 83.3/97.62  |
| CE / Tri | 82.67/97.02 | NULL         | 82.57/97.32 | NULL        | **84.72**/**97.68** |


### Veri-Wild
HAlf Precision - Baseline mAP: 87.24 CMC1: 96.65 lambda:0.8 beta:1.0
Value updated due to weird behaviour with nn.parallel 
| R50      | 4G          | 2G          | 2X2G        | 2X          | 4X          |
| -------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| CE + Tri | 85.16/94.81 | 86.66/96.08 | 86.7/95.52  | 87.64/96.39 | 87.31/96.05 |
| CE / Tri | 84.05/93.41 | 86.11/95.28 | 86.91/95.78 | 87.31/95.98 | 87.73/96.02 |

| R50+BoT  | 4G          | 2G          | 2X2G        | 2X          | 4X         |
| -------- | ----------- | ----------- | ----------- | ----------- | ---------- |
| CE + Tri | 86.07/95.58 | 86.92/96.29 | 87.11/95.62 | 88.57/**96.79** | **88.9**/96.55 |
| CE / Tri | 85.29/94.85 | NULL        | 86.52/95.21 | NULL        | 86.9/95.75 |



### VehicleID 

Half Precision - Baseline mAP:  91.44 CMC1: 86.72 lambda:0.2 beta:1.0


| R50      | 4G          | 2G          | 2X2G        | 2X          | 4X          |
| -------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| CE + Tri | 91.5/86.81  | 91.86/87.35 | 91.6/86.95  | 92.04/87.62 | 91.79/87.28 |
| CE / Tri | 90.68/85.56 | 90.94/85.59 | 91.44/86.66 | 91.45/86.83 | 91.91/87.36 |

Hybrid R50+BoT
| R50+BoT  | 4G          | 2G          | 2X2G        | 2X          | 4X          |
| -------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| CE + Tri | 91.35/86.46 | 91.66/87.01 | 91.48/86.76 | 91.99/87.39 | 92.03/87.49 |
| CE / Tri | 90.36/85.17 | NULL        | 91.15/86.22 | NULL        | **92.32**/**87.97** |


Please cite our paper inspired on proposed techniques or code. ref will be published soon.


Some code as data loading, evaluation, and other is mainly reused from:
Parsing-based View-aware Embedding Network for Vehicle Re-Identification
Bag of Tricks and A Strong Baseline for Deep Person Re-identification
FastREID
