import glob
import os
import re
import cv2
import sys
# First, pass the path of the image
image_path = sys.argv[1]
size = int(sys.argv[2])

try:
    os.mkdir(image_path + '/resized')
except:
    pass

files = glob.glob(image_path + '/*.jpg')
files = files + glob.glob(image_path + '/*.jpeg')

for file in files:
    try:
        print(file)
        image = cv2.imread(file)
        height, width, depth = image.shape
        imgScale = size/width
        newX, newY = image.shape[1]*imgScale, image.shape[0]*imgScale
        image = cv2.resize(image, (int(newX), int(newY)),
                           0, 0, cv2.INTER_LINEAR)

        newfile = image_path + '/resized/' + os.path.basename(file)
        cv2.imwrite(newfile, image)
    except Exception as e:
        print(e, 'Could not load', file)
