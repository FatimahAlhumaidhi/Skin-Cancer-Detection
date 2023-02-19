import os, cv2, numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

def data_pipeline(DIR='classification', BATCHSIZE=8, IMAGESIZE=(256, 256), seed=42, validation_split=0.2):
  trainDataset = image_dataset_from_directory(
              DIR+'/train',
              shuffle=True,
              color_mode="rgb",
              image_size=IMAGESIZE, 
              batch_size=BATCHSIZE
              )
  trainDataset.map(lambda image,label:(Rescaling(scale=1/255.)(image),label))

  validateDataset = image_dataset_from_directory( 
              DIR+'/val',
              shuffle=True,
              color_mode="rgb",
              image_size=IMAGESIZE,
              batch_size=BATCHSIZE
              )
  validateDataset.map(lambda image,label:(Rescaling(scale=1/255.)(image),label))

  testDataset = image_dataset_from_directory( 
              DIR+'/test',
              shuffle=True,
              color_mode="rgb",
              image_size=IMAGESIZE,
              batch_size=BATCHSIZE
              )
  testDataset.map(lambda image,label:(Rescaling(scale=1/255.)(image),label))

  return trainDataset, validateDataset, testDataset


def data_pipeline_with_augmentation(DIR='classification', BATCHSIZE=16, IMAGESIZE=(256, 256), seed=42, validation_split=0.2):

    imageDatagen = ImageDataGenerator(
        # validation_split=validation_split, 
        preprocessing_function=preprocess_input, 
        rescale=1/255.,
        # seed=seed,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    
    trainDataset = imageDatagen.flow_from_directory(
        DIR+'/train',
        class_mode='binary',
        color_mode = 'rgb',
        target_size = IMAGESIZE,
        batch_size = BATCHSIZE,
        # subset = 'training'
    )
    validateDataset = imageDatagen.flow_from_directory(
        DIR+'/val',
        class_mode='binary',
        color_mode = 'rgb',
        target_size = IMAGESIZE,
        batch_size = BATCHSIZE,
        # subset = 'validation'
    )
    testDataset = imageDatagen.flow_from_directory(
        DIR+'/test',
        class_mode='binary',
        color_mode = 'rgb',
        target_size = IMAGESIZE,
        batch_size = BATCHSIZE,
    )
    

    return trainDataset, validateDataset, testDataset 

def segmentation_dataset(imagesPath=os.path.join('HAM10000', 'image', '*'), masksPath=os.path.join('HAM10000', 'mask',  '*'), test_size=0.2, random_state=42):

    Images = glob(imagesPath) 
    Masks = glob(masksPath) 
    Images.sort()
    Masks.sort()

    X = np.zeros((len(Images), 256, 256, 3), dtype=np.float32)
    Y = np.zeros((len(Images), 256, 256, 1), dtype=np.bool_)

    for n, (imagepath, maskpath) in tqdm(enumerate(zip(Images, Masks))):
        img = cv2.imread(imagepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        img = cv2.resize(img, (256, 256))
        mask =  cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256)).reshape((256,256,1))
        
        X[n] = img
        Y[n] = mask

    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    del X, Y, Images, Masks

    return (X_train, y_train), (X_valid, y_valid)


def classification_dataset(path='classification'):

    malignant = glob(os.path.join(path, 'malignant', '*'))
    benign = glob(os.path.join(path, 'benign', '*'))
    Images = malignant + benign 

    X = np.zeros((len(Images), 256, 256, 3), dtype=np.float32)
    Y = np.zeros((len(X),), dtype=np.bool_)

    for n, imagepath in tqdm(enumerate(Images)):
        img = cv2.imread(imagepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        X[n] = img
        if imagepath in malignant:
            Y[n] = 1
        else:
            Y[n] = 0

    return X, Y

def getMasks(path='masks'):

    Masks = glob(path)
    X = np.zeros((len(Masks), 256, 256), dtype=np.bool_)

    for n, Maskpath in tqdm(enumerate(Masks)):
        img = cv2.imread(Maskpath, cv2.IMREAD_GRAYSCALE)
        X[n] = img

    return X

def augmentation_dataset(imagesPath = os.path.join('trainGAN', '*')):
    realImages = glob(imagesPath)

    Images = np.zeros((len(realImages), 256, 256, 3), dtype=np.float32)

    for i, image in tqdm(enumerate(realImages)):
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        Images[i] = img

    return Images


