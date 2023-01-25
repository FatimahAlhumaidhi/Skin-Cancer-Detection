import cv2, os
import numpy as np 
from glob import glob


def preprocess(imagepath, IMAGESIZE = (256, 256), destinationPath=''):
    """
    clean and resize skin leison images using the dull razor algorithm
    Developed by engineer Javier Velasquez (2020)
    """
    image = cv2.imread(imagepath,cv2.IMREAD_COLOR)
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (9, 9)) 
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel) 
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)

    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(bhg, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    _, mask = cv2.threshold(erosion, 15, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA) 

    dst = cv2.resize(dst, IMAGESIZE)
    cv2.imwrite(os.path.join(destinationPath, imagepath.split('\\')[-1]), dst)


def preprocess_mask(maskPath, IMAGESIZE = (256, 256), destinationPath=''):
    """
    resize mask and write it back
    """
    mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
    mask = mask / 255.
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    mask = cv2.resize(mask, IMAGESIZE)
    cv2.imwrite(os.path.join(destinationPath, maskPath.split('\\')[-1]), 255*mask)


def apply_mask(image, mask, imname, destinationPath=''):
    """
    apply mask and write it back
    """
    mask = mask.astype(np.uint8)
    seg = cv2.bitwise_and(image, image, mask=mask)*255.
    newimage = cv2.cvtColor(seg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(destinationPath, imname), newimage)
    return seg



def main():
    imagesPath = os.path.join('dataset', 'image', '*')
    masksPath = os.path.join('dataset',  'mask', '*')

    imageDestination = os.makedirs(os.path.join('cleaned_dataset','image'))
    maskDestination = os.mkdir(os.path.join('cleaned_dataset','mask'))

    images = glob(imagesPath)
    for image in images:
        preprocess(image, destinationPath=imageDestination)
    
    masks = glob(masksPath)
    for mask in masks:
        preprocess_mask(mask, destinationPath=maskDestination)
    

if __name__ == '__main__':
    main()
