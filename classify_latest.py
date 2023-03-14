from infer import infer
import os
import glob
from constants import classes


def classify():
    img_dir = '/mnt/jpegs'
    list_of_files = glob.glob(img_dir+'/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    return infer(latest_file)


if __name__ == '__main__':
    class_index = classify()
    print(classes[class_index])
