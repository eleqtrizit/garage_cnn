from infer import infer
import os
import glob
from constants import classes


def classify():
    img_dir = '/var/cache/zoneminder/events/2'
    list_of_files = glob.glob(img_dir+'/**/*jpg', recursive = True)
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    return infer(latest_file)


if __name__ == '__main__':
    class_index = classify()
    print(classes[class_index])
    
