from imagesearch import config 
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Input 
from tensorflow.keras.models import Model 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.preprocessing.image import load_img 
import matplotlib.pyplot as plt 
import numpy as np 
import cv2 
import os 

rows = open(config.ANNOTS_PATH).read().strip().split('\n')
data =[]
targets = []
filenames = []

for idx, row in enumerate(rows):
    if idx == 0: 
        continue 
    row = row.split(',')
    (filename,_,_,_,xmin,ymin,xmax,ymax) = row 
    imagePath = os.path.sep.join([config.IMAGES_PATH, filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    
    xmin = float(xmin)/w 
    ymin = float(ymin)/h 
    xmax = float(xmax)/w
    ymax = float(ymax)/h 

    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    data.append(image)
    targets.append((xmin, ymin, xmax, ymax))
    filenames.append(filename)

train_data = np.array(data, dtype='float32')/255.0
train_targets = np.array(targets, dtype='float32')

rows = open(config.VAL_ANNOTS_PATH).read().strip().split('\n')
val_data = []
val_targets = []
val_filenames = []

for idx ,row in enumerate(rows):
    if idx == 0:
        continue 
    row = row.split(',')
    (filename, _, _, _, xmin, ymin, xmax, ymax) = row 
    imagePath = os.path.sep.join([config.VAL_PATH, filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]

    xmin = float(xmin)/w 
    ymin = float(ymin)/h 
    xmax = float(xmax)/w 
    ymax = float(ymax)/h 

    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    val_data.append(image)
    val_targets.append((xmin, ymin, xmax, ymax))
    val_filenames.append(filename)

val_data = np.array(val_data, dtype='float32')/255.0
val_targets = np.array(val_targets, dtype='float32')

vgg = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
vgg.trainable = False 

flatten = vgg.output
flatten = Flatten()(flatten)

bbox = Dense(128, activation='relu')(flatten)
bbox = Dense(64, activation='relu')(bbox)
bbox = Dense(32, activation='relu')(bbox)
bbox = Dense(4, activation = 'sigmoid')(bbox)

model = Model(inputs=vgg.input, outputs=bbox)
opt = Adam(config.init_lr)
model.compile(loss='mean_squared_error', optimizer= opt)

H = model.fit(
    train_data, train_targets,
    validation_data = (val_data, val_targets),
    batch_size = config.batch_size,
    epochs = config.num_epochs,
    verbose = 1)

model.save(config.MODEL_PATH, save_format='h5')


N = config.num_epochs
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label = 'val_loss')
plt.xlabel('Epochs #')
plt.ylabel('Loss')
plt.legend(loc='lower left')
plt.savefig(config.PLOT_PATH)

    