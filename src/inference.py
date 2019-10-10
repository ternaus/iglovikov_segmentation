"""Script to create segmented masks."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from catalyst.dl import SupervisedRunner
from catalyst.dl import utils
from iglovikov_helper_functions.config_parsing.from_py import py2cfg
from iglovikov_helper_functions.utils.img_tools import unpad
from pytorch_toolbelt.inference.tta import TTAWrapper, fliplr_image2mask, d4_image2mask
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import TestSegmentationDataset


class ApplySoftmaxToLogits(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x.softmax(dim=1)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-i", "--input_path", type=Path, required=True, help="Path to the target image folder.")
    arg("-j", "--num_workers", type=int, help="Number of CPU threads.", default=12)
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-w", "--checkpoint", type=Path, help="Path to checkpoint.", required=True)
    arg("-b", "--batch_size", type=int, help="Batch_size.", default=1)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("-v", "--visualize", help="If add visualized predictions.", action="store_true")
    return parser.parse_args()


def load_checkpoint(file_path: (Path, str), rename_in_layers: dict = None):
    """Loads pytorch checkpoint, optionally renaming layer names.



    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}


    Returns:

    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if rename_in_layers is not None:
        model_state_dict = checkpoint["model_state_dict"]

        result = {}
        for key, value in model_state_dict.items():
            for key_r, value_r in rename_in_layers.items():
                key = key.replace(key_r, value_r)

            result[key] = value

        checkpoint["model_state_dict"] = result

    return checkpoint


def main():
    args = get_args()
    image_paths = sorted(args.input_path.glob("*.jpg"))

    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    if args.visualize:
        vis_output_path = Path(str(args.output_path) + "_vis")
        vis_output_path.mkdir(exist_ok=True, parents=True)

    test_aug = config.test_augmentations

    model = config.model

    checkpoint = load_checkpoint(args.checkpoint, {"model.0.": "", "model.": ""})
    utils.unpack_checkpoint(checkpoint, model=model)

    model = nn.Sequential(model, ApplySoftmaxToLogits())

    model, _, _, _, device = utils.process_components(model=model)

    if args.tta == "lr":
        model = TTAWrapper(model, fliplr_image2mask)
    elif args.tta == "d4":
        model = TTAWrapper(model, d4_image2mask)

    runner = SupervisedRunner(model=model, device=device)

    with torch.no_grad():
        test_loader = DataLoader(
            TestSegmentationDataset(image_paths, test_aug, factor=config.pad_factor, imread_lib=config.imread_library),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        for input_images in tqdm(test_loader):
            raw_predictions = runner.predict_batch({"features": input_images["features"].cuda()})["logits"]

            image_height, image_width = input_images["features"].shape[2:]

            pads = input_images["pads"].cpu().numpy()

            image_ids = input_images["image_id"]

            _, predictions = raw_predictions.max(1)

            for i in range(args.batch_size):
                unpadded_mask = predictions[i].cpu().numpy()

                if unpadded_mask.shape != (image_height, image_width):
                    unpadded_mask = cv2.resize(
                        unpadded_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST
                    )

                mask = unpad(unpadded_mask, pads[i]).astype(np.uint8)

                mask_name = image_ids[i] + ".png"
                cv2.imwrite(str(args.output_path / mask_name), mask)
                if args.visualize:
                    factor = 255 // config.num_classes
                    cv2.imwrite(str(vis_output_path / mask_name), mask * factor)


if __name__ == "__main__":
    main()
