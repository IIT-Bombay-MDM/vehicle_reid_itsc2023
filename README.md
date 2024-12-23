# vehicle_reid_itsc2023
## Strength in Diversity: Multi-Branch Representation Learning for Vehicle Re-Identification

Implementation of paper [Strength in Diversity: Multi-Branch Representation Learning for Vehicle Re-Identification](https://ieeexplore.ieee.org/document/10422175)

[See original implementation here.](https://github.com/videturfortuna/vehicle_reid_itsc2023)

To train:
```console
python main.py --model_arch MBR_4G --config ./config/config_BoT_VERIWILD.yaml
```

Test (all images):
```console
python teste.py --path_weights ./logs/VERIWILD/MBR_4G/0/
```

Test (single image):
```console
python test.py --path_weights ./logs/VERIWILD/MBR_4G/0/ --query_image_id 5
```

Note that the scripts above do not use FAISS. To use FAISS, see [db.py](db.py).

```console
python db.py --path_weights ./logs/Veri776/MBR_4G/0/ --query_image_id 5
```

The VeRi dataset is available [here](https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset). Some training logs are available [here](https://drive.google.com/file/d/1sMVU9C8pu8O3m0PaNTsYmi2-SPUJ3GxF/view?usp=share_link). The logs contain weights which can be used to test the model.
