from tensorflow.keras.models import load_model
import cv2
import numpy as np 
from skimage.filters import threshold_otsu

segmenter = load_model('models/Unet.hdf5') 
classifier = load_model('models/EfficientNetB7.hdf5') 

def preprocess(image, IMAGESIZE=(256, 256)):

    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (9, 9)) 
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel) 
    bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT) 

    _, mask = cv2.threshold(bhg, 15, 255, cv2.THRESH_BINARY) 
    dst = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA) 

    dst = cv2.resize(dst, IMAGESIZE)
    return dst

def apply_mask(image, mask):

    mask = mask.astype(np.uint8)
    seg = cv2.bitwise_and(image, image, mask=mask)

    return seg
    

def predict(image):
    
    image = preprocess(image)
    predictedMask = segmenter.predict(np.reshape(image, (1, 256, 256, 3))) 
    # _, mask = cv2.threshold(predictedMask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    predictedMask = np.reshape(predictedMask, (256, 256, 1))
    thresh = threshold_otsu(predictedMask)
    mask = predictedMask > thresh
    output = apply_mask(image, mask)
    prediction = classifier.predict(np.reshape(output, (1, 256, 256, 3)))

    return prediction[0] > 0.5
