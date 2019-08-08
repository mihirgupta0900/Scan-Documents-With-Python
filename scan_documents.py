## Document Scanner

# Importing packages
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True,
                help = 'Path to the image to be scanned')
args = vars(ap.parse_args())

## STEP 1 EDGE DETECTION

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
img = cv2.imread(args['image'])
ratio = img.shape[0]/500.0
orig = img.copy()
img = imutils.resize(img, height = 500)

# Convert the image to grayscale,blur it and detect edges
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

#  Show  the original image andthe edge detected image
print('STEP 1: Edge Detection')
cv2.imshow('Image', img)
cv2.imshow('Edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

## STEP 2: FINDING CONTOURS

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea,  reverse = True)[:5]

#loop over CONTOURS
for c in cnts:
    #approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

#show thecontour on the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow('Outline', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Step 3: Apply a Perspective Transform & Threshold

warped = four_point_transform(orig, screenCnt.reshape(4,2) * ratio)

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped,  cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = 'gaussian')
warped = (warped > T).astype('uint8') * 255

# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
