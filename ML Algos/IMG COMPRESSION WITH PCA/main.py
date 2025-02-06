import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



img = cv2.imread('img1.jpg')

cv2.imshow('Image Window', img)

cv2.waitKey(0)

cv2.destroyAllWindows()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('Image Window', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
