import os 

BASE_PATH = 'dataset'
IMAGES_PATH  = os.path.sep.join([BASE_PATH, 'train'])
ANNOTS_PATH = os.path.sep.join([IMAGES_PATH, '_annotations.csv'])

VAL_PATH = os.path.sep.join([BASE_PATH, 'valid'])
VAL_ANNOTS_PATH = os.path.sep.join([VAL_PATH, '_annotations.csv'])

TEST_PATH = os.path.sep.join([BASE_PATH, 'test'])
TEST_ANNOTS_PATH = os.path.sep.join([TEST_PATH, '_annotations.csv'])


BASE_MODEL = 'output'
MODEL_PATH = os.path.sep.join([BASE_MODEL, 'detector.h5'])
PLOT_PATH  = os.path.sep.join([BASE_MODEL, 'plot.png'])
TEST_TXT = os.path.sep.join([BASE_MODEL, 'test.txt'])

init_lr = 1e-4 
num_epochs = 2
batch_size = 32 

