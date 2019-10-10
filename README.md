# Installation

Run `pip install -r requirements.txt` to install all requred libraries.

# Data preparation
## Cityscapes

1. Use [this](https://github.com/ternaus/iglovikov_helper_functions/tree/master/iglovikov_helper_functions/data_processing/cityscapes)
guide to prepare the data.
2. Put prepared data to the `data` folder or modify path to the train and test files in the config file.


# Training
Almost all trainining parameters are specified in the config file. You may check examples in the configs folder.

You may star training, running:

`python -m src.train <path_to_config_file>`

# Inference

Run:

```
python -m src.inference -c <path to config> \
                        -i <path to input images> \
                        -o <output_path> 
                        -w <path to pth file with weights> \
                        -b <batch size> \
                        -j <num_cpu_threads>
```

# Contributing 
To contribute follow [this guide.](docs/contributing.rst)

