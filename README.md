# cv-CAM-based-img-segmentation

## 📖 Overview

This project explores **weakly supervised** learning techniques for image segmentation, focusing primarily on the Grad-CAM++ method ([paper](https://arxiv.org/abs/1710.11063)).

GradCAM++ requires only image-level labels for training, eliminating the need for pixel-level annotations. The project aims to evaluate the performance of Grad-CAM++ on the Oxford Pets dataset, comparing it with fully supervised methods.

In additional, we experimented with **self-training** and **superpixels SLIC** to improve the performance of Grad-CAM++.

## 📊 Results: Test mIOU (in %) on Oxford-IIIT Pet Dataset

| Method             | Backbone Model      | 5 epochs  | 10 epochs | 20 epochs |
|:-------------------|:--------------------|----------:|----------:|----------:|
| SLIC + Grad-CAM++  | DeepLabV3 ResNet-50 | **76.21** | **77.61** | -         |
| Grad-CAM++         | DeepLabV3 ResNet-50 | 73.81     | 74.35     | -         |
| Grad-CAM++         | U-Net               | 67.29     | 68.38     | 71.04     |
| Fully-Supervised   | DeepLabV3 ResNet-50 | 91.78     | 92.29     | -         |
| Fully-Supervised   | U-Net               | 79.20     | 82.90     | 86.56     |

## 🖼️ Visualisation -- Grad-CAM++ + SLIC Superpixels

![gradcam_slic_demo_img](docs/assets/gradcam_slic_demo.png)

## 🧪 Experiments

### Dataset

Dataset used in this project is Oxford Pets dataset. The dataset can be downloaded from the following link:
[Oxford Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

### Models and Generated CAM Masks

Models weights and the generated CAM Masks to reproduce the experiments can be downloaded from the following link:
[Model Weights and CAM Data](https://www.icloud.com/iclouddrive/05f4KsOqKw_NywJZlf6qifJ5g#GradCAM-Weakly-Supervision)

## 🚀 Noteboooks and Scripts

Experiments and results are stored in the `notebooks` directory. Separate scripts are also provided for training and evaluation.

To run the scripts, install all the dependencies listed in `requirements.txt`:

```shell
pip install -r requirements.txt
```

Then, nevigate to the root directory, where the `README.md` file is located, and run the following commands to train and evaluate the models.

### Training

- GradCAM++ deeplabv3 resnet50 with self-training and superpixels slic

    ```shell
    python -m src.train.train_cam_slic_deeplab
    ```

- GradCAM++ deeplabv3 resnet50

    ```shell
    python -m src.train.train_cam_deeplab
    ```

- GradCAM++ Unet

    ```shell
    python -m src.train.train_cam_unet
    ```

- baseline deeplabv3 resnet50 model

    ```shell
    python -m src.train.train_baseline_deeplab
    ```

- baseline Unet model

    ```shell
    python -m src.train.train_baseline_unet
    ```

### Evaluation

- GradCAM++ deeplabv3 resnet50 with self-training and superpixels slic

    ```shell
    python -m src.eval.eval_cam_slic_deeplab
    ```

- GradCAM++ deeplabv3 resnet50

    ```shell
    python -m src.eval.eval_cam_deeplab
    ```

- GradCAM++ Unet

    ```shell
    python -m src.eval.eval_cam_unet
    ```

- baseline deeplabv3 resnet50 model

    ```shell
    python -m src.eval.eval_baseline_deeplab 
    ```

- baseline Unet model

    ```shell
    python -m src.eval.eval_baseline_unet
    ```
