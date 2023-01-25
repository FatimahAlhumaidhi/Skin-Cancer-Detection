import numpy as np
import pandas as pd
from sklearn import *
import imutils, cv2
from tqdm import tqdm 
from scipy import cluster 
from dataloader import classification_dataset


def get_area_primeter(images):

    areas_primeters = np.zeros((len(images), 2), dtype=np.float32)

    for i, img in tqdm(enumerate(images)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        areas = []
        for c in cnts:
            for c in cnts:
                areas.append(cv2.contourArea(c))
        perimeters = []
        for c in cnts:
            for c in cnts:
                perimeters.append(cv2.arcLength(c,True)) 
        areas_primeters[i, :] =  np.array([max(areas, default= -1), max(perimeters, default= -1)])

    return areas_primeters

def colors_extraction(image, NUM_CLUSTERS = 5):

    shape  = image.shape
    image = image.reshape(np.product(shape[:2]), shape[2]).astype(np.float32)
    codes, _ = cluster.vq.kmeans(image, NUM_CLUSTERS)
    vecs, _ = cluster.vq.vq(image, codes)
    counts, _ = np.histogram(vecs, len(codes)) 
    index_max = np.argmax(counts)

    return codes[index_max] 


def extract_features():

    X, Y = classification_dataset()
    X_colormap, _ = classification_dataset(path='colormap')
    dataset = pd.DataFrame(Y, columns = ['Y'])

    areas_primeters = get_area_primeter(X_colormap)
    dataset['area'], dataset['primeter'] = areas_primeters[:, 0], areas_primeters[:, 1]


    top_colors = np.zeros((len(X), 3), dtype=np.float32)
    for i, img in tqdm(enumerate(X)):
        top_colors[i, :]  = colors_extraction(img)

    colors = ['red', 'grean', 'blue']
    for i, color in tqdm(enumerate(colors)):
        dataset[color] = top_colors[:, i]
        dataset[f'mean_{color}'], dataset[f'std_{color}'] = X[:, :, :, i].reshape((len(X), 256*256)).mean(axis=1), X[:, :, :, i].reshape((len(X), 256*256)).std(axis=1) 


    top_valus_hsv = np.zeros((len(X), 3), dtype=np.float32)
    for i, img in tqdm(enumerate(X)):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        top_valus_hsv[i, :]  = colors_extraction(img)

    values = ['hue', 'saturation', 'value']
    for i, value in tqdm(enumerate(values)):
        dataset[value] = top_valus_hsv[:, i]
        dataset[f'mean_{value}'], dataset[f'std_{value}'] = X[:, :, :, i].reshape((len(X), 256*256)).mean(axis=1), X[:, :, :, i].reshape((len(X), 256*256)).std(axis=1) 



    dataset.to_csv('imagesFeatures(sorted).csv', index=False)
    
    dataset = dataset.sample(frac=1)
    dataset.to_csv('imagesFeatures.csv', index=False)

if __name__ == '__main__':
    extract_features()