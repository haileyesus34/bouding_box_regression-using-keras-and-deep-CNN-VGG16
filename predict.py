from imagesearch import config 
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.models import load_model 
import numpy as np 
import mimetypes
import argparse
import imutils
from imutils import paths
import cv2
import os 


imagePaths = []

rows = open(config.TEST_ANNOTS_PATH).read().strip().split('\n')
for idx, row in enumerate(rows):

    if idx == 0: 
        continue
    row = row.split(',')
    filename, _,_,_,_,_,_,_ = row 
    imagePaths.append(filename)

model = load_model(config.MODEL_PATH)
for imagePath in imagePaths: 
    pat = os.path.sep.join([config.TEST_PATH, imagePath])
    print(pat)
    image = load_img(pat, target_size=(224, 224))
    image = img_to_array(image)/255.0
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)[0]
    xmin, ymin, xmax, ymax = preds

    image = cv2.imread(pat)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    xmin = int(xmin*w)
    ymin = int(ymin*h)
    xmax = int(xmax*w)
    ymax = int(ymax*h)

    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    cv2.imshow('output', image)
    cv2.waitKey(0)







