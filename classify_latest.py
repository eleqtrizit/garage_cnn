from infer import infer
import os
import glob

def classify():
        img_dir = '/mnt/jpegs'
        list_of_files = glob.glob(img_dir+'/*')
        latest_file = max(list_of_files, key=os.path.getctime)

        resized = image.resize((640, 360))
        resized.save('/tmp/temp.jpg')