#importing modules
import streamlit as st
import numpy as np
import keras
import keras.utils as image
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

#import model
model = load_model("C:/Users/Admin/Downloads/mod1.hdf5")

#image loder
img = st.file_uploader("Choose an image...", type=("jpeg","img","jpg"))

#main 
imgs = image.load_img(img, target_size=(32,32))
img_tensor = image.img_to_array(imgs)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = img_tensor/.255

#prediction
st.title("The result of prediction is:")

predictions = model.predict(img_tensor)

if(predictions=="1"):
    st.title("Positive")
else:
    st.title("negative")