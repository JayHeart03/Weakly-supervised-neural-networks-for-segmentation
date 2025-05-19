import pathlib
import json

import torch
import numpy as np
import xml.etree.ElementTree as ET

from typing import Tuple
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
from skimage.segmentation import slic
from skimage.morphology import opening, closing, disk


def split_dataset(
	data_dir: pathlib.PurePath,
	train_frac: float = 0.7,
) -> Tuple[Dataset, Dataset]:

	anns_folder = data_dir.joinpath("oxford-iiit-pet", "annotations")

	# Download the dataset if it doesn't exist
	datasets.OxfordIIITPet(root=data_dir, download=True,)

	image_ids = []
	labels = []

	split = ("trainval", "test")
	for s in split:
		with open(anns_folder / f"{s}.txt") as file:
			for line in file:
				image_id, label, _, _ = line.strip().split()
				image_ids.append(image_id)
				labels.append(int(label) - 1)

	n = len(image_ids)
	train_size = int(train_frac * n)

	indices = torch.randperm(n)

	train_idx = indices[:train_size]
	test_idx = indices[train_size:]

	train_ids = [image_ids[i] for i in train_idx]
	train_labels = [labels[i] for i in train_idx]

	test_ids = [image_ids[i] for i in test_idx]
	test_labels = [labels[i] for i in test_idx]

	with open(data_dir / 'train_ids.json', 'w') as f:
		json.dump(train_ids, f)

	with open(data_dir / 'train_labels.json', 'w') as f:
		json.dump(train_labels, f)

	with open(data_dir / 'test_ids.json', 'w') as f:
		json.dump(test_ids, f)

	with open(data_dir / 'test_labels.json', 'w') as f:
		json.dump(test_labels, f)

	return train_ids, train_labels, test_ids, test_labels


def construct_dataset(
	data_dir: pathlib.PurePath,
	train_transforms=None,
	test_transforms=None,
	cam_dir=None,
	raw_cam=False,
) -> Tuple[Dataset, Dataset]:

	# Define the directories for images and segmentation masks
	img_dir = data_dir.joinpath("oxford-iiit-pet", "images")
	seg_dir = data_dir.joinpath("oxford-iiit-pet", "annotations", "trimaps")

	# Download the dataset if it doesn't exist
	dataset = datasets.OxfordIIITPet(root=data_dir, download=True,)

	# Get the class names
	classes = dataset.classes

	with open(data_dir / 'train_ids.json', 'r') as f:
		train_ids = json.load(f)
	with open(data_dir / 'train_labels.json', 'r') as f:
		train_labels = json.load(f)
	with open(data_dir / 'test_ids.json', 'r') as f:
		test_ids = json.load(f)
	with open(data_dir / 'test_labels.json', 'r') as f:
		test_labels = json.load(f)

	# Get the paths to the images and segmentation masks
	train_images = [img_dir / f"{image_id}.jpg" for image_id in train_ids]
	train_segs = [seg_dir / f"{image_id}.png" for image_id in train_ids]
	test_images = [img_dir / f"{image_id}.jpg" for image_id in test_ids]
	test_segs = [seg_dir / f"{image_id}.png" for image_id in test_ids]

	if cam_dir is not None:
		if raw_cam:
			train_cams = [cam_dir / f"{image_id}_graycam.png" for image_id in train_ids]
			test_cams = [cam_dir / f"{image_id}_cam.png" for image_id in test_ids]
		else:
			train_cams = [cam_dir / f"{image_id}_mask.png" for image_id in train_ids]
			test_cams = [cam_dir / f"{image_id}_mask.png" for image_id in test_ids]
		# Create the datasets
		train_dataset = OxfordPetsDataset(train_images, train_segs, train_cams, train_labels, classes, transforms=train_transforms)
		test_dataset = OxfordPetsDataset(test_images, test_segs, test_cams, test_labels, classes, transforms=test_transforms)
	else:
		# Create the datasets without CAMs
		train_dataset = OxfordPetsDataset(train_images, train_segs, None, train_labels, classes, transforms=train_transforms)
		test_dataset = OxfordPetsDataset(test_images, test_segs, None, test_labels, classes, transforms=test_transforms)

	return train_dataset, test_dataset


def construct_dataset_by_bbox(
	data_dir: pathlib.PurePath,
	train_transforms=None,
	test_transforms=None,
) -> Tuple[Dataset, Dataset]:

	# Define the directories for images and segmentation masks
	img_dir = data_dir.joinpath("oxford-iiit-pet", "images")
	seg_dir = data_dir.joinpath("oxford-iiit-pet", "annotations", "trimaps")
	anns_folder = data_dir.joinpath("oxford-iiit-pet", "annotations")
	bbox_folder = data_dir.joinpath("oxford-iiit-pet", "annotations", "xmls")

	# Download the dataset if it doesn't exist
	dataset = datasets.OxfordIIITPet(root=data_dir, download=True,)

	# Get the class names
	classes = dataset.classes

	# Get the path to the images, labels, segmentation masks
	image_ids = []
	labels = []
	has_bbox = []

	split = ("trainval", "test")
	for s in split:
		with open(anns_folder / f"{s}.txt") as file:
			for line in file:
				image_id, label, _, _ = line.strip().split()
				image_ids.append(image_id)
				labels.append(int(label) - 1)
				# Check if the corresponding XML file (bounding box) exists
				xml_path = bbox_folder / f"{image_id}.xml"
				has_bbox.append(xml_path.exists())

	# If split_by_bbox is True, we split the dataset based on the presence of bounding boxes
	# images with bounding boxes will be used for training, and those without will be used for testing
	train_images = [img_dir / f"{image_id}.jpg" for image_id, has_bbox in zip(image_ids, has_bbox) if has_bbox]
	train_bboxs = [bbox_folder / f"{image_id}.xml" for image_id, has_bbox in zip(image_ids, has_bbox) if has_bbox]
	train_labels = [labels[i] for i, has_bbox in enumerate(has_bbox) if has_bbox]

	test_images = [img_dir / f"{image_id}.jpg" for image_id, has_bbox in zip(image_ids, has_bbox) if not has_bbox]
	test_segs = [seg_dir / f"{image_id}.png" for image_id, has_bbox in zip(image_ids, has_bbox) if not has_bbox]
	test_labels = [labels[i] for i, has_bbox in enumerate(has_bbox) if not has_bbox]

	train_dataset = BBoxDataset(train_images, train_bboxs, train_labels, classes, transforms=train_transforms)
	test_dataset = OxfordPetsDataset(test_images, test_segs, None, test_labels, classes, transforms=test_transforms)

	return train_dataset, test_dataset


class OxfordPetsDataset(Dataset):
	"""
	Custom dataset class for the Oxford Pets dataset. 
	This custom dataset returns the image, segmentation mask, and label when called.
	
	Args:
		imgs (list): List of image file paths.
		segs (list): List of segmentation mask file paths.
		labels (list): List of class labels.
		classes (list): List of class names.
		transforms (callable, optional): Transformations to apply to the images and masks.
	"""

	def __init__(self, imgs, segs, cams, labels, classes, transforms=None):
		# These are lists of paths
		self._images = imgs
		self._segs = segs
		self._labels = labels
		self._cams = cams

		self.transforms = transforms
		self.classes = classes

	def __len__(self):
		return len(self._images)

	def __getitem__(self, idx):
		
		###### Read in original images ######
		image_path = self._images[idx]

		# Load the image
		image = Image.open(image_path).convert("RGB")

		###### Read in original labels ######
		# Get the label
		label = self._labels[idx]

		###### Load the segmentation mask ######
		seg_path = self._segs[idx]
		seg = Image.open(seg_path)

		###### Load the CAM pseudo mask ######
		if self._cams is not None:
			cam_path = self._cams[idx]
			cam = Image.open(cam_path).convert("L")
		else:
			cam = None

		# Apply the transformations if given
		if self.transforms:
			image, seg, cam = self.transforms(image, seg, cam)

		# If cam is None, return a zero tensor
		cam = cam if cam is not None else torch.zeros_like(seg)

		return image, seg, cam, label

	def update_pseudo_mask(self, new_masks):
		"""
		Update the pseudo mask for the dataset.
		
		Args:
			new_mask (list): List of new masks to be used for the dataset.
		"""
		self._cams = new_masks


class BBoxDataset(OxfordPetsDataset):

	def __init__(self, imgs, bbox, labels, classes, transforms=None):
		super().__init__(imgs, bbox, None, labels, classes, transforms)
		self.has_pseudo_mask = False
		self.new_masks = []

	def __getitem__(self, idx):

		image_path = self._images[idx]

		# Load the image
		image = Image.open(image_path).convert("RGB")

		# Get the label
		label = self._labels[idx]

		if not self.has_pseudo_mask:

			bbox_path = self._segs[idx]
			bbox = self.create_binary_mask_from_bbox(bbox_path)

		else:
			bbox = self.new_masks[idx]
			bbox = Image.fromarray(bbox)

		if self.transforms:
			image, bbox, _ = self.transforms(image, bbox)

		return image, bbox, label

	def update_pseudo_mask(self, new_masks):
		"""
		Update the pseudo mask for the dataset.
		
		Args:
			new_mask (list): List of new masks to be used for the dataset.
		"""
		self.new_masks = new_masks
		self.has_pseudo_mask = True

	def create_binary_mask_from_bbox(self, xml_path):
		"""
		Create a binary segmentation mask from bounding box coordinates in an XML file.
		The mask has values 0 (background) and 1 (foreground, within bbox).
		
		Args:
			xml_path (Path): Path to the XML file with bbox annotations and size info.
		
		Returns:
			torch.Tensor: Binary mask with values 0 and 1 (shape: [height, width], dtype: torch.uint8).
		"""
		# Parse the XML file
		tree = ET.parse(xml_path)
		root = tree.getroot()
		
		# Get image size from <size> tag
		size = root.find("size")
		if size is None:
			raise ValueError(f"No <size> tag found in {xml_path}")
		width = int(size.find("width").text)
		height = int(size.find("height").text)
		
		# Create a blank binary mask (0 for background)
		mask = np.zeros((height, width), dtype=np.uint8)
		
		# Extract bounding box coordinates
		for obj in root.findall("object"):
			bbox = obj.find("bndbox")
			if bbox is not None:
				xmin = int(bbox.find("xmin").text)
				ymin = int(bbox.find("ymin").text)
				xmax = int(bbox.find("xmax").text)
				ymax = int(bbox.find("ymax").text)
				
				# Ensure coordinates are within image bounds
				xmin = max(0, xmin)
				ymin = max(0, ymin)
				xmax = min(width, xmax)
				ymax = min(height, ymax)
				
				# Set the bounding box area to 1 (foreground)
				mask[ymin:ymax, xmin:xmax] = 1
		
		# Convert to PIL Image
		pil_image = Image.fromarray(mask)
		
		return pil_image


def refine_mask_with_superpixel(
    image_tensor,
    prob_map,
    n_segments: int = 200,
    compactness: float = 0.001,
    sigma: float = 1.0,
    selem_radius: int = None
):
    """
    Refine a per-pixel probability map by superpixel soft-voting, with smoothing.

    Parameters
    ----------
    image_tensor : torch.Tensor
        Input image in CHW format ([3, H, W]).
    prob_map : np.ndarray
        Softmax output of shape (H, W, C) giving per-pixel class probabilities.
    n_segments : int
        Number of superpixels for SLIC.
    compactness : float
        Compactness parameter for SLIC.
    sigma : float
        Gaussian smoothing parameter for SLIC.
    selem_radius : int or None
        Radius for morphological opening+closing. If None, no smoothing is done.

    Returns
    -------
    smoothed : np.ndarray or None
        If `selem_radius` is given, the result after opening+closing, else None.
    refined : np.ndarray
        Hard labels after superpixel soft-voting, shape (H, W), dtype int.
    superpixels : np.ndarray
        The SLIC label map, shape (H, W), dtype int.
    """
    # Convert image tensor to HWC numpy
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    prob = prob_map.permute(1, 2, 0).cpu().numpy()  # Convert to HWC

    # 1) Compute superpixels
    superpixels = slic(
        img,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0
    )
    # print(f"SLIC produced {superpixels.max()+1} superpixels; shape = {superpixels.shape}")

    # 2) Soft‐voting within each superpixel
    H, W, C = prob.shape
    refined = np.zeros((H, W), dtype=np.int32)
    for sp_val in np.unique(superpixels):
        mask = (superpixels == sp_val)
        sp_probs = prob[mask]           # shape: (n_pixels_in_sp, C)
        summed = sp_probs.sum(axis=0)       # shape: (C,)
        refined[mask] = np.argmax(summed)

    # 3) Optional morphological smoothing
    smoothed = None
    if selem_radius is not None:
        selem = disk(selem_radius)
        # Opening then closing to remove small islands & fill gaps
        smooth0 = opening(refined.astype(np.uint8), selem)
        smoothed = closing(smooth0, selem)

    return smoothed, refined, superpixels
