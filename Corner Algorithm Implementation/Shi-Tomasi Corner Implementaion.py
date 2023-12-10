import cv2
import numpy as np
from IPython.display import Image

def shi_tomasi_detect_corner(img_path, maxCornerNB, qualityLevel, minDistance=10):
    r = 0
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, maxCornerNB, qualityLevel, minDistance)
    corners = np.intp(corners)

    for i in corners:
        x, y = i.ravel()

        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        r = r + 1
    
    cv2.imwrite('example_shitomasi.png', img)
    print(r)
    return 'example_shitomasi.png'

img_path = shi_tomasi_detect_corner('Ground_Truth_Dataset/1.png', 300000 , 0.1)
Image(img_path)
