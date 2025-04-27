import streamlit as st
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

# Initialize
if not os.path.exists('uploads'):
    os.makedirs('uploads')

detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except Exception as e:
        print(e)
        return False

def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if results == []:
        return None

    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]
    
    image_face = Image.fromarray(face)
    image_face = image_face.resize((224, 224))

    face_array = np.asarray(image_face)
    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features):
    similarity = cosine_similarity(features.reshape(1, -1), feature_list)
    index_pos = np.argmax(similarity)
    return index_pos

# Streamlit UI
st.title('✨ Which Bollywood Celebrity Are You? ✨')

uploaded_image = st.file_uploader('Upload an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    with st.spinner('Processing...'):
        if save_uploaded_image(uploaded_image):
            # Load and display uploaded image
            display_image = Image.open(uploaded_image)
            st.image(display_image, caption='Uploaded Image', use_column_width=True)

            # Extract features
            features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
            
            if features is None:
                st.error("No face detected. Please upload a clear face image!")
            else:
                # Recommend
                index_pos = recommend(feature_list, features)
                predicted_actor = filenames[index_pos].split(os.sep)[-2]  # Get folder name (actor name)pip install tf-nightly


                # Show prediction
                st.success(f"Looks like you resemble **{predicted_actor}**! 🎬")

                col1, col2 = st.columns(2)
                with col1:
                    st.header('Your Image')
                    st.image(display_image)
                with col2:
                    st.header('Matched Celebrity')
                    st.image(filenames[index_pos], width=300)
