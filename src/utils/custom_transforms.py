"""
This script contains custom image transformations for segmentation tasks. 
Since v2.Transform class is not available in this torchvision version, 
we have to define the transforms manually for segmentation.

The code is suggested by Pytorch team in the blog post:
https://pytorch.org/blog/extending-torchvisions-transforms-to-object-detection-segmentation-and-video-tasks/

The reference code can be found in:
https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
"""

import numpy as np
import torch

from torchvision import transforms as T
from torchvision.transforms import functional as F


class FormatCAM:
    """
    Convert a raw CAM (torch.Tensor with values in [0, 1])
    into a colourful PIL Image using a jet colour palette.
    """
    def __init__(self, isRaw):
        self.isRaw = isRaw

    def __call__(self, image, seg, cam):

        if self.isRaw:
            cam = cam.numpy()
            cam = cam / 255.0 # normalise to [0, 1]
            cam = torch.from_numpy(cam)

        cam = cam.squeeze(0) if cam.ndim == 3 else cam
        cam = cam.long()
        return image, seg, cam

class Compose:
    """
    Custom Compose class that suggested by Torchvision reference code.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None, cam=None):

        for t in self.transforms:
            image, target, cam = t(image, target, cam)

        return image, target, cam

class ConvertMaskToBinary:
    """
    Given a mapping, convert the trimap mask to binary.

    Args:
        mapping (dict): A dictionary that maps the original mask values to binary values.
                        Default is {1: 1, 2: 0, 3: 1}, which means:
                        1 object body -> object (1)
                        2 background -> background (0)
                        3 object border -> object (1)
    """
    def __init__(self, mapping={1: 1, 2: 0, 3: 1}):
        self.mapping = mapping

    def __call__(self, image, target, cam=None):
        for k, v in self.mapping.items():
            target[target == k] = v

        return image, target, cam

class ResizeImgAndMask:
    """
    Custom Resize class that resizes images and masks to a specified size.
    This custom transform the image and mask using different interpolation methods.
    Args:
        size (tuple): The target size to resize the image and mask.
        img_interpolation (int): Interpolation method for images. Default is InterpolationMode.BILINEAR.
        mask_interpolation (int): Interpolation method for masks. Default is InterpolationMode.NEAREST.
    """
    def __init__(self, size, img_interpolation=T.InterpolationMode.BILINEAR, mask_interpolation=T.InterpolationMode.NEAREST):
        self.size = size
        self.img_interpolation = img_interpolation
        self.mask_interpolation = mask_interpolation

    def __call__(self, image, target=None, cam=None):
        # For images, use bilinear interpolation with antialiasing
        image = F.resize(image, self.size, antialias=True, interpolation=self.img_interpolation)

        if target is None:
            return image

        # For masks, use nearest neighbor interpolation
        target = F.resize(target, self.size, interpolation=self.mask_interpolation, antialias=False)

        if cam is not None:
            cam = F.resize(cam, self.size, interpolation=self.mask_interpolation, antialias=False)
            return image, target, cam
        
        return image, target, cam

class RandomRotation:
    """
    Custom RandomRotation class that rotates images and masks by a random angle. 
    This class ensure that the image and mask are rotated by the same angle.

    Args:
        degrees (int):  The maximum rotation angle in degrees. 
                        The rotation angle will be randomly chosen from the range [-degrees, degrees].
    """
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, target, cam=None):
        angle = torch.randint(-self.degrees, self.degrees + 1, (1,)).item()
        image = F.rotate(image, angle)
        target = F.rotate(target, angle)
        
        if cam is not None:
            cam = F.rotate(cam, angle)
            return image, target, cam

        return image, target, cam

class RandomHorizontalFlip:
    """
    Custom RandomHorizontalFlip class that suggested by Torchvision reference code.
    """
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, cam=None):
        if torch.rand(1) < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
            cam = F.hflip(cam) if cam is not None else None

        return image, target, cam


class RandomVerticalFlip:
    """
    Custom RandomVerticalFlip class that suggested by Torchvision reference code.
    """
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, cam=None):
        if torch.rand(1) < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
            cam = F.vflip(cam) if cam is not None else None

        return image, target, cam


class PILToTensor:
    """
    Custom PILToTensor class that suggested by Torchvision reference code.
    """
    def __call__(self, image, target=None, cam=None):
        image = F.pil_to_tensor(image)
        
        if target is None:
            return image

        #################################
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        # Ensure the target is a 3D tensor
        target = torch.unsqueeze(target, dim=0) if target.ndim == 2 else target
        #################################
        if cam is not None:
            cam = torch.as_tensor(np.array(cam), dtype=torch.uint8)
            # Ensure the cam is a 3D tensor
            cam = torch.unsqueeze(cam, dim=0) if cam.ndim == 2 else cam

        return image, target, cam

class ToDtype:
    """
    Custom ToDtype class that suggested by Torchvision reference code.
    """
    def __init__(self, dtype, scale=False):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, image, target=None, cam=None):
        if target is None:
            return image.to(dtype=self.dtype)

        if not self.scale:
            return image.to(dtype=self.dtype), target
        image = F.convert_image_dtype(image, self.dtype)
        
        target = target.squeeze(0) if target.ndim == 3 else target

        return image, target, cam

class Normalize:
    """
    Custom Normalize class that suggested by Torchvision reference code.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, cam=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target, cam


### v2 Transform is not available in the specified torchvision version :(

# import torch

# from typing import Any, Dict
# from torchvision import tv_tensors
# from torchvision.transforms import v2


# class CustomResize(v2.Transform):
#     """
#     Custom transform class that resizes images and masks to a specified size. 
#     This custom transform the image and mask using different interpolation methods.

#     Args:
#         size (tuple): The target size to resize the image and mask.
#         img_interpolation (int): Interpolation method for images. Default is InterpolationMode.BILINEAR.
#         mask_interpolation (int): Interpolation method for masks. Default is InterpolationMode.NEAREST.
#     """

#     def __init__(
#         self, 
#         size, 
#         img_interpolation=v2.InterpolationMode.BILINEAR, 
#         mask_interpolation=v2.InterpolationMode.NEAREST,
#         binary_mask=True,
#     ):
#         super().__init__()
#         self.size = size
#         self.img_resize = v2.Resize(self.size, interpolation=img_interpolation)
#         self.mask_resize = v2.Resize(self.size, interpolation=mask_interpolation)
#         self.binary_mask = binary_mask

#     def transform(self, inpt: Any, params: Dict[str, Any]):
#         """
#         Apply the transform to the input image or mask.
#         Args:
#             inpt (Any): The input image or mask to be transformed.
#             params (Dict[str, Any]): Additional parameters for the transform. Not used in this case.
#         Returns:
#             Any: The transformed input.
#         """

#         # If the input is an image, apply the image resize transform
#         if isinstance(inpt, tv_tensors.Image):
#             inpt = self.img_resize(inpt)
#         # If the input is a mask, apply the mask resize transform
#         elif isinstance(inpt, tv_tensors.Mask):
#             if self.binary_mask:
#                 inpt[inpt == 3] = 1     # Take the border and body to be the positive class
#                 inpt[inpt == 2] = 0     # Take the background as the negative class
#             inpt = self.mask_resize(inpt)
#             inpt = inpt.to(torch.int64)
#         # Else do not apply any transform
#         else:
#             print(f"No resize is performed on {type(inpt)}")
        
#         return inpt
