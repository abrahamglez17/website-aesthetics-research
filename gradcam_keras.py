from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import h5py

# Definir la arquitectura del modelo
def create_model():
    model = Sequential()
    model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv1', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='pool1'))

    model.add(Conv2D(256, (5, 5), activation='relu', padding='same', name='conv2'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='pool2'))

    model.add(Conv2D(384, (3, 3), activation='relu', padding='same', name='conv3'))

    model.add(Conv2D(384, (3, 3), activation='relu', padding='same', name='conv4'))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2), name='pool5'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dense(1, activation='sigmoid', name='fc8_flickr'))  # Ajustar según la arquitectura original

    return model

# Crear el modelo
model = create_model()

# Cargar los pesos manualmente
model_path = 'pretrained-models/flickr_style.h5'
with h5py.File(model_path, 'r') as f:
    for layer in model.layers:
        if layer.name in f.keys():
            layer_weights = f[layer.name]
            weights = [layer_weights[weight_name][:] for weight_name in layer_weights.keys()]
            layer.set_weights(weights)

print("Pesos cargados exitosamente.")

def generate_gradcam(model, img_path, layer_name='conv5'):
    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]

    # Obtener la última capa convolucional
    last_conv_layer = model.get_layer(layer_name)
    grads = tf.keras.backend.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = tf.keras.backend.mean(grads, axis=(0, 1, 2))

    iterate = tf.keras.backend.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(pooled_grads_value.shape[-1]):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Superponer la Grad-CAM en la imagen original
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite('gradcam.jpg', superimposed_img)

# Ruta a la imagen
img_path = 'datasets/rating-based-dataset/images/english/3.png'
generate_gradcam(model, img_path)
