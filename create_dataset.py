import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

data_dir = './data'

for dir in os.listdir(data_dir):
    for image_path in os.listdir(os.path.join(data_dir, dir))[:1]:
        image = cv2.imread(os.path.join(data_dir, dir, image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure()
        plt.imshow(image_rgb)

plt.show()