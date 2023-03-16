
from pathlib import Path
from platform import platform

from common import get_score, process_image, show_collage
from constants import MAX_IMAGES, sort_directories

MAJOR_CHANGE_SCORE = 0.67
COLLAGE_SCORE_THRESHOLD = 0.8


def process_input(images_to_move: list, response: str) -> None:
    """
    Takes a list of images and a response from the user and moves the images to the appropriate location.
    If the user responds 'd' the images are deleted, otherwise they are moved to the location specified by the number
    of the response.
    :param images_to_move: list of images to move
    :param response: user's response, either 'd' or a number
    :return: None
    """
    if response == 'd':
        print("Deleting images")
        for image in images_to_move:
            image.unlink()
    else:
        target_folder = sort_directories[int(response)]
        print(f"Moving images to {target_folder}")
        for image in images_to_move:
            target = target_folder / image.name
            print(target)
            image.rename(target)


def get_input():
    print("Sort previous images to which directory?")
    for i, sort_directory in enumerate(sort_directories):
        print(f'{i}: {sort_directory}')
    print('D: Delete images')
    return input(":")


def main(dir: str) -> None:
    path = Path(dir)
    files = path.glob('*.jpg')
    file = files.__next__()
    img1, collage_image1 = process_image(file)
    images_to_move = [file]
    collage_images = [collage_image1]

    for file in files:
        img2, collage_image2 = process_image(file)

        score = get_score(img1, img2)
        print("Image matching score between the two images:", score)

        if score < MAJOR_CHANGE_SCORE or len(collage_images) == MAX_IMAGES:
            print("Major change detected" if score < MAJOR_CHANGE_SCORE else "MAX_IMAGES images reached")
            show_collage(collage_images)
            process_input(images_to_move, get_input())
            images_to_move = [file]
            collage_images = [collage_image2]
        else:
            images_to_move.append(file)

            # add only slightly questionable photos to the collage, but keep at least two images
            if score < COLLAGE_SCORE_THRESHOLD or len(collage_images) < 2:
                collage_images.append(collage_image2)
            else:
                # replace the last image in the collage with the current image
                collage_images[-1] = collage_image2
        img1 = img2


if __name__ == '__main__':
    if 'Windows' in platform():
        main('P:/Jpegs/')
    else:
        print('This script is only for Windows')
