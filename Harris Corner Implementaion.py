import cv2
import numpy as np
from scipy import ndimage as ndi
import os
from google.colab.patches import cv2_imshow

from google.colab import drive
drive.mount("/content/drive", force_remount=True)
img_path = "/content/drive/My Drive/Colab Notebooks/corner/"

k = 0.04
window_size = 5
threshold = 10000000.00

for name in os.listdir(img_path):
    if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        input_img = cv2.imread(img_path + name, 0)

        if input_img is not None:
            print("Image loaded successfully.")
            print("Image shape:", input_img.shape)
            cv2_imshow(input_img)
            cv2.waitKey(0)
        else:
            print("Error loading image:", name)

        corner_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_GRAY2RGB)
        edge_img = cv2.cvtColor(input_img.copy(), cv2.COLOR_GRAY2RGB)

        offset = int(window_size/2)
        y_range = input_img.shape[0] - offset
        x_range = input_img.shape[1] - offset

        dy, dx = np.gradient(input_img)

        Ixx = ndi.gaussian_filter(dx**2, sigma=1)
        Ixy = ndi.gaussian_filter(dy*dx, sigma=1)
        Iyy = ndi.gaussian_filter(dy**2, sigma=1)

        corner_detected = 0
        edge_detected = 0

        for y in range(offset, y_range):
            for x in range(offset, x_range):
                start_y = y - offset
                end_y = y + offset + 1
                start_x = x - offset
                end_x = x + offset + 1

                windowIxx = Ixx[start_y: end_y, start_x: end_x]
                windowIxy = Ixy[start_y: end_y, start_x: end_x]
                windowIyy = Iyy[start_y: end_y, start_x: end_x]

                Sxx = windowIxx.sum()
                Sxy = windowIxy.sum()
                Syy = windowIyy.sum()

                det = (Sxx * Syy) - (Sxy**2)
                trace = Sxx + Syy

                r = det - k*(trace**2)

                if r > threshold:
                    corner_img[y, x] = (0, 0, 255)
                    corner_detected += 1
                elif r < 0:
                    edge_img[y, x] = (0, 0, 255)
                    edge_detected += 1

        print(f"Image {name} - Corners: {corner_detected}, Edges: {edge_detected}")

        cv2_imshow(corner_img)
        cv2.waitKey(0)

        cv2_imshow(edge_img)
        cv2.waitKey(0)