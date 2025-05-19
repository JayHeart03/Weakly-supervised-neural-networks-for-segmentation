import pathlib
import torch

import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
from src import utils
from ..utils import get_root_dir


if __name__ == "__main__":
    # Specify the model to be evaluated
    model_name = "self_train_deeplab_epoch_5.pth"

    # Set Hyperparameters
    IMAGE_SIZE = 256
    NUM_CLASSES = 3

    TRAIN_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Configure paths
    ROOT = get_root_dir(pathlib.Path(__file__).resolve(), anchor="README.md")
    DATA_DIR = ROOT.joinpath("data")
    MODEL_DIR = ROOT.joinpath("models")
    OUTPUT_DIR = ROOT.joinpath("output")

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
    ])

    test_transforms = utils.Compose([
        utils.PILToTensor(),
        utils.ResizeImgAndMask(size=(IMAGE_SIZE, IMAGE_SIZE)),
        utils.ConvertMaskToBinary(),
        utils.ToDtype(dtype=torch.float32, scale=True),
    ])

    # Load dataset
    trainset, testset = utils.construct_dataset(
        data_dir=DATA_DIR,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
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

    # Load model
    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.load_state_dict(torch.load(MODEL_DIR.joinpath(model_name), weights_only=True))
    model.to(device)

    # Evaluate model
    model.eval()
    num_test_samples = len(testloader.dataset)
    miou = 0
    with torch.no_grad():
        for images, segs, _, _ in tqdm(testloader):
            images = images.to(device)
            segs = segs.to(device)
            batch_size = images.size(0)

            outputs = model(images)["out"]
            preds = (torch.argmax(outputs, dim=1) > 0)

            miou += utils.mean_iou(preds, segs, num_classes=2) * batch_size

        miou /= num_test_samples
        print(f"Model: {model_name},Mean IoU: {miou:.4f}")
