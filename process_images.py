import itertools
import random
from pathlib import Path
from platform import platform

import cv2

from common import (center_crop, center_crop_x, center_crop_y, directories,
                    rotate_image, standard_processing)

SAMPLE_FACTOR = 250
sorted_base_dir = Path('P:/sorted/')
processed_base_dir = Path('P:/processed/')
sorted_base_dir.mkdir(exist_ok=True)
sort_directories = [sorted_base_dir / directory for directory in directories]
train_directories = [processed_base_dir / "train" / f"{directory}" for directory in directories]
test_directories = [processed_base_dir / "test" / f"{directory}"for directory in directories]
sorted_base_dir = Path('P:/sorted/')
sorted_base_dir.mkdir(exist_ok=True)


class Rotate:
    angles = list(range(-15, 16))

    def __call__(self, image):
        return rotate_image(image, random.choices(self.angles, k=1)[0])


class CenterCrop:
    xys = list(itertools.product(range(center_crop_x), range(center_crop_y)))

    def __call__(self, image):
        x, y = random.choices(self.xys, k=1)[0]
        return center_crop(image, x, y)


class FileSampler:
    def __init__(self, file_list):
        self.file_list = file_list

    def __call__(self):
        return str(random.choices(self.file_list, k=1)[0])


def file_count(sort_dir):
    sort_path = Path(sort_dir)
    return len(list(sort_path.glob('*.jpg')))


def process(sort_dir, train_dir, test_path, max_samples):
    sort_path = Path(sort_dir)
    train_path = Path(train_dir)
    test_path = Path(test_path)
    rotator = Rotate()
    center_cropper = CenterCrop()
    files = sort_path.glob('*.jpg')
    file_sampler = FileSampler(list(files))

    for i in range(max_samples):
        img = standard_processing(file_sampler())

        rotated_image = rotator(img)
        centered_image = center_cropper(rotated_image)

        if i % 10 != 0:
            file_name = str(train_path / f"{random.randint(0, 100000000)}.jpg")
        else:
            file_name = str(test_path / f"{random.randint(0, 100000000)}.jpg")
        print(file_name)
        cv2.imwrite(file_name, centered_image)
        # cv2.imshow('image', centered_image)
        # cv2.waitKey(1)


def main():
    max_samples = min(file_count(sort_dir) for sort_dir in sort_directories) * SAMPLE_FACTOR
    for sort_dir, train_dir, test_dir in zip(sort_directories, train_directories, test_directories):
        train_dir.mkdir(exist_ok=True, parents=True)
        test_dir.mkdir(exist_ok=True, parents=True)
        process(sort_dir, train_dir, test_dir, max_samples)


if __name__ == '__main__':
    if 'Windows' in platform():
        main()
    else:
        print('This script is only for Windows')
