import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.inception_v3 import decode_predictions


def load_inception_model():
    model = InceptionV3(include_top=True, weights='imagenet')
    return model


def calculate_is(images, model):
    all_preds = []

    for img_path in images:
        img = image.load_img(img_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        preds = tf.nn.softmax(preds).numpy()
        all_preds.append(preds)

    all_preds = np.vstack(all_preds)

    p_y = np.mean(all_preds, axis=0)

    epsilon = 1e-10
    kl_divergence = np.sum(all_preds * (np.log(all_preds + epsilon) - np.log(p_y + epsilon)), axis=1)
    is_score = np.exp(np.mean(kl_divergence))

    return is_score


generated_images = ["../new_image/0/8_round/2_8/0.png", "../new_image/0/8_round/4_8/1.png", "../new_image/0/8_round/3_8/1.png", "../new_image/0/8_round/1_8/1.png"]
model = load_inception_model()
is_value = calculate_is(generated_images, model)
print("Inception Score:", is_value)
