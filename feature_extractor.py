import os
import pickle
from tqdm import tqdm
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image


# Step 1: Create a list of all image file paths
dataset_path = 'dataset'
actors = os.listdir(dataset_path)

filenames = []

for actor in actors:
    actor_folder = os.path.join(dataset_path, actor)
    for file in os.listdir(actor_folder):
        filenames.append(os.path.join(actor_folder, file))

pickle.dump(filenames, open('filenames.pkl', 'wb'))

# Step 2: Extract features
model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')

def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

features = []

for file in tqdm(filenames):
    features.append(feature_extractor(file, model))

pickle.dump(features, open('embedding.pkl', 'wb'))
