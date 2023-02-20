import numpy as np
from pydicom import read_file
from skimage import morphology
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import math

def transform_to_hu(file_path):
    # load dcm file
    ds = read_file(file_path)
    # load image
    image = ds.pixel_array
    # get intercept, slope, and transform
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    hu_image = image * slope + intercept

    return hu_image


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image


def remove_noise(file_path, display=False):
    hu_image = transform_to_hu(file_path)
    brain_image = window_image(hu_image, 40, 80)

    segmentation = morphology.dilation(brain_image, np.ones((1, 1)))
    labels, label_nb = ndimage.label(segmentation)

    label_count = np.bincount(labels.ravel().astype(np.int))
    label_count[0] = 0

    mask = labels == label_count.argmax()

    # Improve the brain mask
    mask = morphology.dilation(mask, np.ones((1, 1)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))

    masked_image = mask * brain_image

    if display:
        plt.figure(figsize=(15, 2.5))
        plt.subplot(141)
        plt.imshow(brain_image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(142)
        plt.imshow(mask)
        plt.title('Mask')
        plt.axis('off')

        plt.subplot(143)
        plt.imshow(masked_image)
        plt.title('Final Image')
        plt.axis('off')

    return masked_image


def remove_tilt(masked_image, display = False):


    img=np.uint8(masked_image)
    img_copy = np.pad(img, ((106, 106), (106, 106)), 'constant', constant_values=(0,0))
    contours, hier =cv2.findContours (img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask=np.zeros(img_copy.shape, np.uint8)

    # find the biggest contour (c) by the area
    c = max(contours, key = cv2.contourArea)

    (x,y),(MA,ma),angle = cv2.fitEllipse(c)

    #cv2.ellipse(img, ((x,y), (MA,ma), angle), color=(0, 255, 0), thickness=2)

    rmajor = max(MA,ma)/2
    if angle > 90:
        angle -= 90
    else:
        angle += 90
    xtop = x + math.cos(math.radians(angle))*rmajor
    ytop = y + math.sin(math.radians(angle))*rmajor
    xbot = x + math.cos(math.radians(angle+180))*rmajor
    ybot = y + math.sin(math.radians(angle+180))*rmajor
    #cv2.line(img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 255, 0), 3)
    

    M = cv2.getRotationMatrix2D((x, y), angle-90, 1)  #transformation matrix

    img_rot = cv2.warpAffine(img_copy, M, (img_copy.shape[1], img_copy.shape[0]), cv2.INTER_CUBIC)

    #OKMM = cv2.invertAffineTransform(M)
    #img_back = cv2.warpAffine(img_rot,OKMM,(img_copy.shape[1], img_copy.shape[0]), cv2.INTER_CUBIC)
    #img_back = img_back[106:-106, 106:-106] # crop image back to orginal size

    if display:
        plt.figure(figsize=(15, 4))
        plt.subplot(151)
        plt.imshow(masked_image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(152)
        plt.imshow(img_rot)
        plt.title('Rotated Image')
        plt.axis('off')

        #plt.subplot(153)
        #plt.imshow(img_back)
        #plt.title('Zurück gedrehtes Bild')
        #plt.axis('off')

    return img_rot, M



def crop_image(image, display=False):
    # Create a mask with the background pixels
    mask = image == 0

    # Find the brain area
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    
    # Remove the background
    croped_image = image[top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1]]
    
    return croped_image, top_left, bottom_right



def add_pad(image, new_height=512, new_width=512):
    height, width = image.shape

    final_image = np.zeros((new_height, new_width))

    pad_left = int((new_width - width) // 2)
    pad_top = int((new_height - height) // 2)
    
    
    # Replace the pixels with the image's pixels
    final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
    
    return final_image



     

