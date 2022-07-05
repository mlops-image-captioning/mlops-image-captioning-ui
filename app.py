from pathlib import Path
import streamlit as st
from PIL import Image
import hashlib
import pandas as pd
import tensorflow as tf
import io
import numpy as np    
from PIL import Image

SAVE_IMAGE_PATH = Path('./images')
SAVE_IMAGE_PATH.mkdir(exist_ok=True)
LOG_PATH = Path('./log.csv')
LOG_PATH.touch(exist_ok=True)

@st.experimental_memo
def get_model():
    return tf.keras.models.load_model('model.h5')

def generate_caption(image):
    labels = ['T-shirt/top',  'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
         'Sneaker', 'Bag', 'Ankle boot']
    model = get_model()
    img = Image.open(io.BytesIO(image))
    # width, height = img.size
    img = img.convert('L')
    img = img.resize((28, 28))
    # arr = np.asarray(img).reshape((1, width, height, 3))
    arr = np.asarray(img).reshape((1, 28, 28))
    prediction = model.predict(arr)
    prediction = np.argmax(prediction)
    return str(labels[prediction])

st.write('# Image Captioning')

uploaded_image = st.file_uploader("Upload an Image")
if uploaded_image is not None:
    # To read file as bytes:
    image_bytes = uploaded_image.getvalue()

    # image = Image.open('sunrise.jpg')
    st.image(image_bytes)
    generated_caption = generate_caption(image_bytes)
    st.caption(generated_caption)

    md5_hash = hashlib.md5(image_bytes).hexdigest()

    with open(SAVE_IMAGE_PATH / md5_hash, 'wb') as f:
        f.write(image_bytes)
        f.close()

    df = pd.read_csv('log.csv', header=None, names=['image', 'generated_caption', 'reported_caption']).set_index('image')
    df.loc[md5_hash, 'generated_caption'] = generated_caption
    # with open("log.csv", "a") as f:
    #     f.write(f"{md5_hash}\t{generated_caption}\n")

    with st.expander("Report"):
        with st.form("my_form"):
            reported_caption_text = st.text_input('Thanks for reporting! Please enter the caption that the system should have generated.', generated_caption)

            submitted = st.form_submit_button("Submit")
            if submitted:
                # with open("log.csv", "a") as f:
                #     f.write(f"{md5_hash}\t{generated_caption}\t{reported_caption_text}\n")
                df.loc[md5_hash, 'reported_caption'] = reported_caption_text
                st.write('Thanks for reporting!')
    df.to_csv(LOG_PATH, header=False)
