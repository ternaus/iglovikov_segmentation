import argparse
from collections import OrderedDict
from pathlib import Path

from catalyst.dl.callbacks import CheckpointCallback
from catalyst.dl.runner import SupervisedWandbRunner as SupervisedRunner
from iglovikov_helper_functions.config_parsing.from_py import py2cfg
from pytorch_toolbelt.inference.tta import TTAWrapper, d4_image2mask, fliplr_image2mask
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SegmentationDataset


def get_samples(mode: str, config) -> list:
    """Return image path mask path pairs

    Args:
        mode: train or val
        config: general config

    Returns: [(image_path, mask_path)]

    """
    if mode not in ["train", "val"]:
        raise ValueError(f"Only train and val modes are supported, but got {mode}.")

    if mode == "train":
        image_path = config.train_image_path
        mask_path = config.train_mask_path
    else:
        image_path = config.val_image_path
        mask_path = config.val_mask_path

    image_files = sorted(image_path.glob("*.jpg"))

    result = []

    for file_name in tqdm(image_files):
        file_id = Path(file_name).stem

        result += [(file_name, mask_path.joinpath(file_id + ".png"))]

    return result


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-j", "--num_workers", type=int, help="Number of CPU threads.", default=12)
    arg("-r", "--checkpoint_path", type=Path, default=None, help="Path to checkout.")
    return parser.parse_args()


def main():
    args = get_args()

    config = py2cfg(args.config_path)

    train_batch_size = config.train_parameters.train_batch_size
    val_batch_size = config.train_parameters.val_batch_size
    model = config.model

    train_samples = get_samples("train", config)
    val_samples = get_samples("val", config)

    train_aug = config.train_augmentations

    val_aug = config.val_augmentations

    if config.train_parameters.tta == "lr":
        model = TTAWrapper(model, fliplr_image2mask)
    elif config.train_parameters.tta == "d4":
        model = TTAWrapper(model, d4_image2mask)

    train_loader = DataLoader(
        SegmentationDataset(
            train_samples,
            train_aug,
            num_samples=config.num_samples,
            downsample_mask_factor=config.train_parameters.downsample_mask_factor,
        ),
        batch_size=train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        SegmentationDataset(val_samples, val_aug),
        batch_size=val_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    data_loaders = OrderedDict()
    data_loaders["train"] = train_loader
    data_loaders["valid"] = valid_loader

    callbacks = config.callbacks
    if args.checkpoint_path is not None:
        callbacks += [CheckpointCallback(resume=args.checkpoint_path)]

    # model training
    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=config.loss,
        optimizer=config.optimizer,
        callbacks=callbacks,
        logdir=config.logdir,
        loaders=data_loaders,
        num_epochs=config.train_parameters.num_epochs,
        scheduler=config.scheduler,
        verbose=True,
        minimize_metric=True,
        fp16=config.train_parameters.fp16,
    )


if __name__ == "__main__":
    main()
