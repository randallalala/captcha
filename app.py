import cv2
import pytesseract
import numpy as np

from pytesseract import Output

# export PATH="/path/to/dir:$PATH"
# C:\Users\randa\AppData\Local\Tesseract-OCR

image = cv2.imread('image3.jpg')
image = image.astype("uint8")

# WHITELIST
# custom_config = r'--oem 3 --psm 6'
custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 --psm 6'
# ----
# BLACKLIST
# custom_config = r'-c tessedit_char_blacklist=0123456789 --psm 6'

#OPTION 1 - VAR RUNNING, KEEP IN ORDER
# get_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# remove_noise = cv2.medianBlur(get_grayscale, 5)
# closing = cv2.morphologyEx(remove_noise, cv2.MORPH_CLOSE, None)
# kernel = np.ones((5, 5), np.uint8) 
# opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
# thresholding = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# d = pytesseract.image_to_string(thresholding, config=custom_config)
# print(d)

#OPTION 1 - VAR RUNNING, KEEP IN ORDER
get_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
remove_noise = cv2.medianBlur(get_grayscale, 5)
closing = cv2.morphologyEx(remove_noise, cv2.MORPH_CLOSE, None)
kernel = np.ones((5, 5), np.uint8) 
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
thresholding = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
d = pytesseract.image_to_string(thresholding, config=custom_config)
# print(type(d))
print(str(d.strip()))

#OPTION 2 - FUNCTIONS
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)
def canny(image):
    return cv2.Canny(image, 100, 200)
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
# cannyy = canny(image)
# dilate = dilate(cannyy)
# deskeww = deskew(image)
# match_template(image, template)
# e = pytesseract.image_to_string(dilate, config=custom_config)
# print(e)

