import numpy as np
import cv2
import k_means.Img_Segmentation.K_means_ImgSeg as IS

test_img = cv2.imread("test.jpg")

k_points = np.random.randint(0, 255, (4, 3))

test = IS.img_seg(test_img, k_points, -1)

cv2.imshow("Test", test)
cv2.waitKey(0)
