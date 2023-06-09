import concurrent.futures
import itertools
import random
from pathlib import Path
from platform import platform

import cv2

from common import center_crop, rotate_image, standard_processing
from constants import center_crop_x, center_crop_y, sort_directories, test_directories, train_directories

SAMPLE_FACTOR = 250


class Rotate:
    angles = list(range(-8, 9))

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
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=32)

    for i in range(max_samples):
        executor.submit(do_transforms, test_path, train_path, rotator, center_cropper, file_sampler, i)
        # do_transforms(test_path, train_path, rotator, center_cropper, file_sampler, i)

    # wait for all threads to finish
    executor.shutdown(wait=True)


def do_transforms(test_path, train_path, rotator, center_cropper, file_sampler, i):
    img = standard_processing(file_sampler())

    img = rotator(img)
    img = center_cropper(img)

    if i % 10 != 0:
        file_name = str(train_path / f"{random.randint(0, 100000000)}.jpg")
    else:
        file_name = str(test_path / f"{random.randint(0, 100000000)}.jpg")
    print(file_name)
    cv2.imwrite(file_name, img)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)


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
