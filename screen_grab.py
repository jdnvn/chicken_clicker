import numpy as np
from PIL import ImageGrab
import cv2

TOP_X = 4
TOP_Y = 51
BOTTOM_X = 600
BOTTOM_Y = 430

count = 297
# while True:
screenshot =  ImageGrab.grab(bbox = (TOP_X, TOP_Y, BOTTOM_X, BOTTOM_Y))
count += 1
screenshot.save(f'negative/img_{count}.png')
# cv2.imshow('image', screenshot_numpy)
# cv2.waitKey(0)
# if cv2.waitKey(25) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
#     break
