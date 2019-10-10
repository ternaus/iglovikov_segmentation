import albumentations as albu
import cv2
import numpy as np
import torch
from iglovikov_helper_functions.utils.img_tools import load_rgb, load_grayscale, pad
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Segmentations Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        samples: pairs of (image_path, mask_path)
        transform (A.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        class_id: class_id to use. If None - use all.
        num_samples: number of samples in epoch
        imread_lib: Library to read images cv2 or jpeg4py
    """

    def __init__(
            self, samples: list, transform, class_id=None, num_samples=None, imread_lib="cv2", downsample_mask_factor=1
    ):
        self.samples = samples
        self.transform = transform
        self.class_id = class_id
        self.num_samples = num_samples
        self.imread_lib = imread_lib
        self.downsample_mask_factor = downsample_mask_factor

    def __len__(self) -> int:
        if self.num_samples is None:
            return len(self.samples)

        return self.num_samples

    def __getitem__(self, idx):
        idx = idx % len(self.samples)

        image_path, mask_path = self.samples[idx]
        image_id = image_path.stem

        image = load_rgb(image_path, lib=self.imread_lib)
        mask = load_grayscale(mask_path)

        # apply augmentations
        sample = self.transform(image=image, mask=mask)
        image, mask = sample["image"], sample["mask"]

        if self.downsample_mask_factor is not None and self.downsample_mask_factor != 1:
            mask_height, mask_width = mask.shape[:2]

            new_mask_height = mask_height // self.downsample_mask_factor
            new_mask_width = mask_width // self.downsample_mask_factor

            mask = cv2.resize(mask, (new_mask_width, new_mask_height), interpolation=cv2.INTER_NEAREST)

        if self.class_id is not None:
            mask = torch.unsqueeze(torch.from_numpy(mask == self.class_id), 0).float()
        else:
            mask = torch.from_numpy(mask).long()

        return {"image_id": image_id, "features": tensor_from_rgb_image(image), "targets": mask}


class TestSegmentationDataset(Dataset):
    """Segmentations Dataset for inference. Read images, apply normalization transformations.

    Args:
        image_paths: pairs of image_path
        transform: data transformation pipeline
        factor: Image size should be divisible by factor. If not we pad it.
        imread_lib: library used to read images. Supported cv2 and jpeg4py
    """

    def __init__(self, image_paths: list, transform: albu.Compose, factor=None, imread_lib="cv2"):
        self.image_paths = image_paths
        self.transform = transform
        self.factor = factor
        self.imread_library = imread_lib

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = load_rgb(image_path, lib=self.imread_library)

        # apply transformations
        normalized_image = self.transform(image=image)["image"]

        if self.factor is not None:
            normalized_image, pads = pad(normalized_image, factor=self.factor)

            return {
                "image_id": image_path.stem,
                "features": tensor_from_rgb_image(normalized_image),
                "pads": np.array(pads),
            }
        else:
            return {"image_id": image_path.stem, "features": tensor_from_rgb_image(normalized_image)}
