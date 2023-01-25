import os, cv2, numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import image_dataset_from_directory


def segmentation_dataset(imagesPath = os.path.join('colormap', 'image', '*'), masksPath = os.path.join('colormap', 'mask',  '*'), test_size=0.2):

    Images = glob(imagesPath) 
    Masks = glob(masksPath) 
    Images.sort()
    Masks.sort()

    X = np.zeros((len(Images), 256, 256, 3), dtype=np.uint8)
    Y = np.zeros((len(Images), 256, 256, 1), dtype=np.bool_)

    for n, (imagepath, maskpath) in tqdm(enumerate(zip(Images, Masks))):
        img = cv2.imread(imagepath)
        mask =  cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
        
        X[n] = img
        Y[n] = mask

    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=test_size, random_state=42) 

    del X, Y, Images, Masks

    return (X_train, y_train), (X_valid, y_valid)



def classification_dataset_batched(DIR, BATCHSIZE=16, IMAGESIZE=(256, 256), seed=42, validation_split=0.2): 
  trainDataset = image_dataset_from_directory(
              DIR,
              validation_split=validation_split,
              subset="training",
              seed=seed,
              image_size=IMAGESIZE, 
              batch_size=BATCHSIZE
              )

  validateDataset = image_dataset_from_directory( 
              DIR,
              validation_split=validation_split,
              subset="validation",
              seed=seed,
              image_size=IMAGESIZE,
              batch_size=BATCHSIZE
              )
  
  return trainDataset, validateDataset 


def classification_dataset(path):
    malignant = glob(os.path.join(path, 'malignant', '*')) 
    benign = glob(os.path.join(path, 'benign', '*')) 
    Images = malignant + benign 

    X = np.zeros((len(Images), 256, 256, 3), dtype=np.uint8) 
    Y = np.zeros((len(X),), dtype=np.bool_) 

    for n, imagepath in tqdm(enumerate(Images)): 
        X[n] = cv2.imread(imagepath)
        if imagepath in malignant:
            Y[n] = 1
        else:
            Y[n] = 0 

    return X, Y


def augmentation_dataset(imagesPath = os.path.join('trainGAN', '*')):
    realImages = glob(imagesPath)

    Images = np.zeros((len(realImages), 256, 256, 3), dtype=np.uint8)

    for i, image in tqdm(enumerate(realImages)):
        img = cv2.imread(image)
        Images[i] = img

    return Images
