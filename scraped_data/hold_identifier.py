# /usr/bin/env 
'''
@file hold_identifier.py
retrieve image of several rock climbing holds and divides
into subimages containing one hold each in new subdirectory

Usage:
python3 hold_identifier.py absolute/filepath/to/images

This will create a new subdirectory within the passed in directory containing the identified holds.
'''
import cv2 as cv
import matplotlib as plt
import numpy as np
import os
import signal
import sys

default_size = 400 # 400x400

# from: https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python/54823710#54823710
def crop_rect(img, rect):
    # get the parameter of the small rectangle
    center = rect[0]
    length = rect[1][0] + 20
    width = rect[1][1] + 20
    size = (length if length < 400 else 400, width if width < 400 else 400)
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    # print("width: {}, height: {}".format(width, height))

    M = cv.getRotationMatrix2D(center, angle, 1)
    img_rot = cv.warpAffine(img, M, (width, height))

    img_crop = cv.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot

# from: https://stackoverflow.com/questions/58248121/opencv-python-how-to-overlay-an-image-into-the-centre-of-another-image
def overlay_images(top, bot):
    h, w = top.shape[0], top.shape[1]
    hh, ww = bot.shape[0], bot.shape[1]

    yoff = round((hh-h)/2)
    xoff = round((ww-w)/2)

    result = bot.copy()
    result[yoff:yoff+h, xoff:xoff+w] = top
    
    return result

if __name__ == '__main__':
    dir_string = "holds_images" # default

    if (len(sys.argv) == 2):
        dir_string = sys.argv[1]   
    else:
        print("#-- Usage: ")
        print("#-- python3 hold_identifier.py absolute/filepath/to/images")
        sys.exit(0)

    directory = os.fsencode(dir_string)
    print(directory)
    identified_holds_folder = f"{dir_string}//identified_holds"
    exists = os.path.isdir(identified_holds_folder)
    if not exists:
        os.mkdir(identified_holds_folder)
        print(f"Creating identified_holds folder in {dir_string}")

    for f in os.listdir(directory):
        img_name = f"{dir_string}//{os.fsdecode(f)}"
        if os.path.isdir(img_name):
            continue
        print(img_name)
        image = cv.imread(img_name)
        if image.shape[0] > default_size or image.shape[1] > default_size:
            largest = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[1]
            scale_percent = (1.0 / (largest / default_size)) * 100
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            image = cv.resize(image, dim, interpolation = cv.INTER_AREA)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 30, 200)
        # find contours w Canny filter
        contours, hierarchy = cv.findContours(edges,
                              cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # temp = image
        # cv.drawContours(temp, contours, -1, (0,255,0), 3)
        # cv.imshow("test", temp)

        # draw rectangles
        for cnt, i in zip(contours, np.arange(len(contours))):
            if cv.contourArea(cnt) < 100:
                continue
            rect = cv.minAreaRect(cnt)
            if rect[1][0] * rect[1][1] < 500:
                continue

            img_cropped, img_rot = crop_rect(image, rect)
            blank = np.zeros([default_size,default_size,3],dtype=np.uint8)
            blank.fill(255)
            # print(f"cropped: {img_cropped.shape} orig: {image.shape} rect: ({rect[1][0]}, {rect[1][1]})")
            img_cropped = overlay_images(img_cropped, blank)
            # cv.imshow("please", img_cropped)
            cv.imwrite(f"{dir_string}//identified_holds//{os.path.splitext(os.fsdecode(f))[0]}_{i}.jpg", 
                       img_cropped)


        # cv.imshow('Contours', image)
        cv.waitKey(0)
        cv.destroyAllWindows()