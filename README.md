Neural Networks (LTAT.02.001)

# Motion prediction model for self driving


## Download dataset

The dataset has to be downloaded inside "input/" folder in the root location of the repo. Download the dataset from [here](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/data?select=scenes)

```
input/
  +- scenes/
        +- sample.zarr
        +- train.zarr
        +- validate.zarr
        +- test.zarr
  +- aerial_map/
        +- aerial_map.png
  +- semantic_map/
        +- semantic_map.pb
  +- meta.json

```

## Install Dependencies

```shell
pip install l5kit
pip install timm
```

## Training

Following backbone CNNs have been added with their respective variants

* [Resnets]()
* [Mixnets]()

1- Open ```notebook_train.ipynb``` and follow step by step instructions to train either unimodal or multimodal model.

2- After training use ```notebook_test.ipynb``` to evaluate your model, like the training notebook all the instructions are added in this file as well.


## References
