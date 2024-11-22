import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image

def load_inception_model():
    model = InceptionV3(include_top=False, pooling='avg')
    return model

def get_features(img_paths, model):
    features = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
        feature = model.predict(img_array)
        features.append(feature)

    features = np.array(features)
    features = features.reshape(features.shape[0], -1)
    return features

def calculate_fid(real_images, generated_images):
    model = load_inception_model()


    real_features = get_features(real_images, model)
    generated_features = get_features(generated_images, model)


    mu_r, sigma_r = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_g, sigma_g = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r.dot(sigma_g), disp=False)
    fid = np.sum(diff**2) + np.trace(sigma_r + sigma_g - 2*covmean)
    return fid

real_images = ["../dataset/Cifar100/train/0/image_14.png", "../dataset/Cifar100/train/8/image_8.png"]
generated_images = ["../new_image/0/0_round/6_0/1.png", "../new_image/0/0_round/6_0/0.png"]
fid_value = calculate_fid(real_images, generated_images)
print("FID:", fid_value)