import streamlit as st
import os
import random
from PIL import Image
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt


# Function to load a random image from a folder
def load_random_image(folder_path):
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    random_image_path = random.choice(images)
    return Image.open(random_image_path)

# Path to your images folder
folder_path = 'data/'

# Streamlit app
st.title('Image Classifier - Real or Fake')

# Allow users to upload an image
uploaded_image = st.file_uploader("Upload an image for classification", type=["png", "jpg", "jpeg"])

# Create two columns
col1, col2 = st.columns(2)

# Display the uploaded image or a random image
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1.image(image, caption='Uploaded Image', use_column_width=True)
else:
    # Display a random image from the folder if no image is uploaded
    if 'image_path' not in st.session_state or st.button('Load Random Image'):
        st.session_state.image_path = load_random_image(folder_path)
    col1.image(st.session_state.image_path, caption='Random Image', use_column_width=True)

# Classify button
if st.button('Classify'):
    # This example uses a fixed classification result.
    # You can replace this part with your actual model prediction logic.
    pipe = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection", device=-1)
    
    if uploaded_image is not None:
        classification_results = pipe(image)
    else:
        classification_results = pipe(st.session_state.image_path)
    
    
    # Convert the classification results to a DataFrame
    df_results = pd.DataFrame(classification_results)
    
    # Plotting
    fig, ax = plt.subplots()
    ax.bar(df_results['label'], df_results['score'], color=['blue', 'orange'])
    ax.set_ylabel('Scores')
    ax.set_title('Classification Scores')
    plt.tight_layout()
    
    # Display the bar chart in Streamlit
    col2.pyplot(fig)
    
    # Load a new random image for next classification if no image is uploaded
    if uploaded_image is None:
        st.session_state.image_path = load_random_image(folder_path)
