# Import necessary libraries
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import scipy

import tensorflow as tf
from tensorflow.keras.applications import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.utils import *
# import pydot

from sklearn.metrics import *
from sklearn.model_selection import *
import tensorflow.keras.backend as K

from tqdm import tqdm, tqdm_notebook
from colorama import Fore
import json
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from skimage.io import *
import time
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import numpy as np 
from tqdm import tqdm
import cv2
import os
import shutil
import itertools
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping

init_notebook_mode(connected=True)
RANDOM_SEED = 123

# Data directories
train_dir = 'Data/TRAIN_CROP'
val_dir = 'Data/VAL_CROP'

# Load data
def load_data():
    train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input)
    
    test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


    train_generator = train_datagen.flow_from_directory(
    train_dir,
    color_mode='rgb',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    seed=RANDOM_SEED)


    validation_generator = test_datagen.flow_from_directory(
    val_dir,
    color_mode='rgb',
    target_size=(224,224),
    batch_size=16,
    class_mode='binary',
    seed=RANDOM_SEED)
    
    return train_generator, validation_generator 

# Define and train the model
def train_model():
    model =tf.keras.models.load_model('model.keras')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    train_data, val_data = load_data()
    model.fit(train_data, validation_data=val_data, epochs=1)
    return model

# Save the model
if __name__ == "__main__":
    model = train_model()
    os.makedirs("models", exist_ok=True)
    model.save("models/brain_tumor_model.h5")  # Saves the model to models/brain_tumor_model.h5
