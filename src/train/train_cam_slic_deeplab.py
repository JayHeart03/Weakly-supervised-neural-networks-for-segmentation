import pathlib
import json
import torch
import utils


import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
from tqdm import tqdm
from src import utils


if __name__ == "__main__":
    # Set Hyperparameters
    IMAGE_SIZE = 256
    NUM_CLASSES = 3

    TRAIN_EPOCHS = 10
    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    NUM_WORKERS = 4

    # Configure paths
    ROOT = utils.get_root_dir(pathlib.Path(__file__).resolve(), anchor="README.md")
    DATA_DIR = ROOT.joinpath("data")
    MODEL_DIR = ROOT.joinpath("models")
    OUTPUT_DIR = ROOT.joinpath("output")
    CAM_DIR = DATA_DIR.joinpath("cam_dataset")

    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = True if torch.cuda.is_available() else False
    print(f"Using device: {device}")

    # Define transforms
    train_transforms = utils.Compose([
        utils.PILToTensor(),
        utils.ResizeImgAndMask(size=(IMAGE_SIZE, IMAGE_SIZE)),
        utils.ConvertMaskToBinary(),
        utils.RandomHorizontalFlip(flip_prob=0.5),
        utils.RandomVerticalFlip(flip_prob=0.5),
        utils.RandomRotation(degrees=30),
        utils.ToDtype(dtype=torch.float32, scale=True),
        utils.FormatCAM(isRaw=False),
    ])
    test_transforms = utils.Compose([
        utils.PILToTensor(),
        utils.ResizeImgAndMask(size=(IMAGE_SIZE, IMAGE_SIZE)),
        utils.ConvertMaskToBinary(),
        utils.ToDtype(dtype=torch.float32, scale=True),
        utils.FormatCAM(isRaw=False),
    ])
    superpixel_transforms = utils.Compose([
        utils.PILToTensor(),
        utils.ResizeImgAndMask(size=(IMAGE_SIZE, IMAGE_SIZE)),
        utils.ToDtype(dtype=torch.float32, scale=True),
    ])

    # Load dataset
    trainset, testset = utils.construct_dataset(
        data_dir=DATA_DIR,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        cam_dir=CAM_DIR,
        raw_cam=False,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        testset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    with open(DATA_DIR / "train_ids.json", "r") as f:
        train_ids = json.load(f)

    # Load model
    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )

    # Train model
    model.train()
    num_train_samples = len(trainloader.dataset)

    for epoch in range(TRAIN_EPOCHS):
        print(f"Epoch {epoch + 1}/{TRAIN_EPOCHS}")
        miou = 0
        # Configure the path where new pseudo masks are saved
        save_dir = DATA_DIR / f"superpixel_refined_mask_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)

        ### Generate refined masks for training
        model.eval()
        trainloader.dataset.transforms = superpixel_transforms
        with torch.no_grad():
            for i in tqdm(range(len(train_ids))):

                image, _, _, _ = trainloader.dataset[i]
                image = image.to(device)

                output = model(image.unsqueeze(0))["out"].cpu().squeeze(0)

                # Refine masks with superpixels and some smoothing
                better_seg, _, _  = utils.refine_mask_with_superpixel(
                    image.cpu(),
                    output,
                    n_segments=200,
                    compactness=0.01,
                    sigma=5.0,
                    selem_radius=3
                )

                image_id = train_ids[i]
                # Save masks as PNGs
                Image.fromarray(better_seg.astype(np.uint8), mode="L").save(save_dir / f"{image_id}_mask.png")

        # Update pseudo masks to be used in training
        new_mask = [save_dir / f"{image_id}_mask.png" for image_id in train_ids]
        trainloader.dataset.update_pseudo_mask(new_mask)

        ### Train model on the refined masks
        model.train()
        trainloader.dataset.transforms = train_transforms
        for i, (images, _, cams, _) in enumerate(tqdm(trainloader)):
            images = images.to(device)
            cams = cams.to(device)
            batch_size = images.size(0)

            # Forward pass
            outputs = model(images)["out"]
            loss = loss_fn(outputs, cams)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            miou += utils.mean_iou(pred, cams, num_classes=NUM_CLASSES) * batch_size

        miou /= num_train_samples
        print(f"Epoch [{epoch + 1}/{TRAIN_EPOCHS}], Loss: {loss.item():.4f}, mIoU: {miou:.4f}")

        # Save model checkpoint
        save_path = MODEL_DIR.joinpath(f"self_train_deeplab_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved at {save_path}")
