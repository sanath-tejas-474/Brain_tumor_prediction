
import streamlit as st
import keras.utils as image
import numpy as np

# import your image classification model and any other necessary modules
import tensorflow as tf
from tensorflow.keras.models import load_model

# load your trained model
model = load_model("C:/Users/Admin/Downloads/mod.hdf5")

def classify_image(image):
  # preprocess the image and classify it using your model
  img = image.load_img(uploaded_file)
  img_tensor = image.img_to_array(img)
  img_tensor = np.expand_dims(img_tensor, axis=0)
  img_tensor = img_tensor/.255
  preds = model.predict(img_tensor)
  
  # return the predicted class and its probability
  return preds[0].argmax(), preds[0].max()

def main():
  # create a file uploader widget
  uploaded_file = st.file_uploader("Choose an image...", type=("jpeg","img","jpg"))
  
  if uploaded_file is not None:
    # convert the uploaded file to a PIL image and classify it
    images = PIL.Image.open(uploaded_file)
    class_name, probability = classify_image(image)
    
    # display the predicted class and probability
    st.write("Predicted class:", class_name)
    st.write("Probability:", probability)
    
    # display the image
    st.image(image, caption="Uploaded image", use_column_width=True)

if __name__ == '__main__':
    main()
