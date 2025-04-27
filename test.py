import cv2
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
detector = MTCNN()

# Testing image
img_path = 'sample/satya.jpg'
sample_img = cv2.imread(img_path)
results = detector.detect_faces(sample_img)

if results:
    x, y, width, height = results[0]['box']
    face = sample_img[y:y+height, x:x+width]

    image_face = Image.fromarray(face)
    image_face = image_face.resize((224,224))

    face_array = np.asarray(image_face).astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()

    similarity = cosine_similarity(result.reshape(1,-1), feature_list)
    index_pos = np.argmax(similarity)

    matched_img = cv2.imread(filenames[index_pos])
    cv2.imshow('Matched Celebrity', matched_img)
    cv2.waitKey(0)
else:
    print("No face detected!")
