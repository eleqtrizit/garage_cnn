import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity

from constants import (FX, FY, bot_crop, center_crop_x, center_crop_y, classes,
                       top_crop)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, )),
     transforms.Grayscale(1)])


class ModelClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(5, 5))
        self.fc1 = nn.Linear(1005, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def verify_size_match(img1: np.ndarray, img2: np.ndarray) -> None:
    # Check to see if the images are the same size.  If not, exit the script
    # with an error message.
    if img1.shape != img2.shape:
        print("Image shape mismatch, skipping.")
        print(f"Image shape: {img1.shape}")
        print(f"Image shape: {img2.shape}")
        print("Images must be the same size.  Please fix manually.")
        exit(1)


def get_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculates the structural similarity between two images.

    Args:
        img1: A numpy array containing the first image.
        img2: A numpy array containing the second image.

    Returns:
        A float indicating the structural similarity between the two images.
    """
    verify_size_match(img1, img2)
    try:
        score = structural_similarity(img1, img2)
    except ValueError:
        score = 0
    return score


def process_image(file: Path) -> np.ndarray:
    """
    This function takes a file path and returns a processed image. This function
    is used to process images in the dataset. The function reads the image,
    resizes it, converts it to grayscale, and crops it.
    """
    image = cv2.imread(str(file))
    image = cv2.resize(image, (0, 0), fx=FX, fy=FY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image[top_crop:-bot_crop, :], image


def find_nearest_sqrt(number: int) -> int:
    """Return the smallest square number greater than or equal to the given number."""
    while math.ceil(math.sqrt(number)) != math.floor(math.sqrt(number)):
        number += 1
    return number


def create_collage(images: np.ndarray) -> np.ndarray:
    # Create a collage from a list of images.
    # The function will try to make a square collage, with as many images as possible.
    # The images are expected to be numpy arrays of shape (H, W, 3).
    collage_size = find_nearest_sqrt(len(images))
    images.extend([np.zeros_like(images[0])] * (collage_size ** 2 - len(images)))
    images_horizontal = []
    horizontals = []
    for _ in range(collage_size):
        if images:
            images_horizontal.extend(images.pop() for _ in range(collage_size))
            horizontals.append(np.hstack(images_horizontal) if images_horizontal else [])
            images_horizontal = []
    return np.vstack(horizontals)


def show_collage(collage_images: np.ndarray, waitkey: int = 1) -> None:
    cv2.imshow("difference", create_collage(collage_images))
    cv2.waitKey(waitkey)


def standard_processing(image: np.ndarray) -> np.ndarray:
    """Standard image processing pipeline.

    Args:
        image: The image to process.

    Returns:
        The processed image.
    """
    img = cv2.imread(image)
    # shrink image
    img = cv2.resize(img, (0, 0), fx=FX, fy=FY)
    # turn image black and white
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # crop out the top and bottom (windows/sky + time/date)
    return img[top_crop:-bot_crop, :]


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    # Rotates an image by a given angle using OpenCV
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))


def center_crop(image: np.ndarray, x: int, y: int) -> np.ndarray:
    # Crop the image from the center
    height, width = image.shape
    center_width = x + width - center_crop_x
    center_height = y + height - center_crop_y
    return image[y:center_height, x:center_width]
