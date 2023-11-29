import tensorflow as tf
import numpy as np


class PredictImage:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        new_model = tf.keras.models.load_model('./models/leak_img_1.model')
        return new_model

    def predict(self, path):
        img = tf.keras.utils.load_img(path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_batch)
        predicted_class = (prediction > 0.5).astype("int32")
        prediction_formatted = 'No Leak' if predicted_class == 1 else 'Leak'

        return {"Prediction": prediction_formatted}
