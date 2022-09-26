import os
import numpy as np
import cv2
from detection.dataset_tools.image_transforming import expo

def apply_clahe(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced_img


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


images_dir = r'F:\TMK\csv1_comets_2_24_08_2022\polarization'

img_files = os.listdir(images_dir)
img_files.sort()

for img_file in img_files:
    img = cv2.imread(os.path.join(images_dir, img_file))
    img = cv2.resize(img, (800, 800))

    img = expo(img, 15)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (41, 41), 0)

    cv2.imshow("gray", gray)
    cv2.imshow("blurred", blurred)

    corrected = np.zeros(gray.shape, dtype=gray.dtype)
    corrected = gray / blurred / 5
    # for i in range(gray.shape[0]):
    #     for j in range(gray.shape[1]):
    #         corrected[i][j] = max(0, int(gray[i][j] / blurred[i][j] * 40))
    
    cv2.imshow(f"corrected", corrected)
    
    if cv2.waitKey() == 27:
        break











