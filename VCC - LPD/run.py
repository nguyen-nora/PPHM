import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

# Load the model
model = load_model('my_model.keras')

# Load the image
img_path = 'TEST1.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict the image
y_cnn = model.predict(x)

# Visualize the detection
plt.figure(figsize=(10, 10))
plt.axis('off')
ny = y_cnn[0] * 255  # Assuming y_cnn[0] contains the bounding box coordinates
img_copy = x[0].copy()  # Copy the image to avoid overwriting
img_copy = cv2.rectangle(img_copy, (int(ny[0]), int(ny[1])), (int(ny[2]), int(ny[3])), (0, 255, 0))
plt.imshow(img_copy.astype('uint8'))
plt.show()
