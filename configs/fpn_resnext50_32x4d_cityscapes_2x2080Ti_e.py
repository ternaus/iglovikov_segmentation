# Cityscapes. 2 x 2080 Ti

from pathlib import Path

import albumentations as albu
import cv2
import segmentation_models_pytorch as smp
from torch.optim import SGD
from pytorch_toolbelt.losses import FocalLoss
import torch

ignore_index = 255

num_classes = 19

encoder_type = "resnext50_32x4d"

preprocess_parameters = smp.encoders.get_preprocessing_params(encoder_type)

mean = preprocess_parameters["mean"]
std = preprocess_parameters["std"]

num_gpu = 2

num_samples = None

train_parameters = dict(
    lr=0.01,
    train_batch_size=8 * num_gpu,
    val_batch_size=num_gpu,
    fp16=False,
    num_epochs=300,
    height_crop_size=512,
    width_crop_size=512,
    ignore_index=ignore_index,
    tta=None,  # can be None, d4 or lr
    downsample_mask_factor=None,  # can be 4 for FPN
)

if train_parameters["downsample_mask_factor"] is not None:
    if not train_parameters["height_crop_size"] / train_parameters["downsample_mask_factor"]:
        raise ValueError(
            f"Height crop size ({train_parameters['height_crop_size']}) "
            f"should be divisible by the downsample_mask_factor "
            f"({train_parameters['downsample_mask_factor']})"
        )

    if not train_parameters["width_crop_size"] / train_parameters["downsample_mask_factor"]:
        raise ValueError(
            f"Width crop size ({train_parameters['width_crop_size']}) "
            f"should be divisible by the downsample_mask_factor"
            f"({train_parameters['downsample_mask_factor']})"
        )

    final_upsampling = None
else:
    final_upsampling = 4

model = smp.FPN(
    encoder_type,
    encoder_weights="imagenet",
    classes=num_classes,
    activation=None,
    final_upsampling=final_upsampling,
    dropout=0.5,
    decoder_merge_policy="cat",
)
pad_factor = 64
imread_library = "cv2"  # can be cv2 or jpeg4py

optimizer = SGD(
    [
        {"params": model.decoder.parameters(), "lr": train_parameters["lr"]},
        # decrease lr for encoder in order not to permute
        # pre-trained weights with large gradients on training start
        {"params": model.encoder.parameters(), "lr": train_parameters["lr"] / 100},
    ],
    weight_decay=1e-4,
    nesterov=True,
    momentum=0.9,
    lr=0.01,
)

normalization = albu.Normalize(mean=mean, std=std, p=1)

train_augmentations = albu.Compose(
    [
        albu.RandomSizedCrop(
            min_max_height=(
                int(0.5 * (train_parameters["height_crop_size"])),
                int(2 * (train_parameters["height_crop_size"])),
            ),
            height=train_parameters["height_crop_size"],
            width=train_parameters["width_crop_size"],
            w2h_ratio=1.0,
            p=1,
        ),
        albu.HorizontalFlip(p=0.5),
        normalization,
    ],
    p=1,
)

val_augmentations = albu.Compose(
    [
        albu.PadIfNeeded(
            min_height=1024, min_width=2048, border_mode=cv2.BORDER_CONSTANT, mask_value=ignore_index, p=1
        ),
        normalization,
    ],
    p=1,
)

test_augmentations = albu.Compose([normalization], p=1)

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda epoch: (1 - (epoch / train_parameters["num_epochs"]) ** 0.9)
)

train_image_path = Path("data/train/images")
train_mask_path = Path("data/train/masks")

val_image_path = Path("data/val/images")
val_mask_path = Path("data/val/masks")

loss = FocalLoss(ignore_index=ignore_index)

callbacks = []

logdir = f"runs/2_2080Ti_{model.name}_e/baseline"