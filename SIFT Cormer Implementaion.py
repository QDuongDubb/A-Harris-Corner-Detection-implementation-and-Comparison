import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def SiftAlgorithm(img1):
  sift = cv.xfeatures2d.SIFT_create()
  kp1, des1 = sift.detectAndCompute(img1, None )
  print(" Total Keypoints img: ", len(kp1))

  keypoint_img1 = np.copy(img1)

  cv.drawKeypoints(img1, kp1,keypoint_img1,flags = 2)

  bf = cv.BFMatcher(cv.NORM_L1, crossCheck = True)

  return keypoint_img1

if __name__ == "__main__":
  img1 = cv.imread("Ground_Truth_Dataset/1.png")

  imgRGB1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

  img1_keypoint = SiftAlgorithm(img1)

  fig = plt.figure(figsize = (16,9))
  ax1 = fig.subplots (1)
  ax1.imshow(imgRGB1)
  ax1.set_title("Anh goc")
  ax1.axis("off")

  fig = plt.figure(figsize = (16, 9))
  ax1 = fig.subplots(1)
  ax1.imshow(img1_keypoint)
  ax1.set_title("Anh va Keypoints: ")
  ax1.axis("off")

  plt.show()