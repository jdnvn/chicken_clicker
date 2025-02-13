import cv2 as cv
import numpy as np
import os
from PIL import ImageGrab
from vision import Vision
import pyautogui
import random
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# we should just capture the whole window
TOP_X = 4
TOP_Y = 51
BOTTOM_X = 600
BOTTOM_Y = 430

MODEL_FOLDER = 'model_4'

cascade = cv.CascadeClassifier(f'{MODEL_FOLDER}/cascade.xml')
vision = Vision(None)

while True:
  time.sleep(10)
  screenshot = ImageGrab.grab(bbox=(TOP_X, TOP_Y, BOTTOM_X, BOTTOM_Y))
  screenshot_numpy = np.array(screenshot, dtype='uint8')
  screenshot_numpy = cv.cvtColor(screenshot_numpy, cv.COLOR_BGR2RGB)
  rectangles = cascade.detectMultiScale(screenshot_numpy, minNeighbors=10, maxSize=(50,50))

  # BOT EXECUTION
  rando = random.choice(rectangles)
  pyautogui.click(rando[0] + rando[2]/2 + TOP_X, rando[1] + rando[3]/2 + TOP_Y)
  # pyautogui.click(rando[0] + rando[2]/2 + TOP_X + 2, rando[1] + rando[3]/2 + TOP_Y + 2)
  # pyautogui.click(rando[0] + rando[2]/2 + TOP_X - 2, rando[1] + rando[3]/2 + TOP_Y - 2)
  # pyautogui.click(rando[0] + rando[2]/2 + TOP_X + 2, rando[1] + rando[3]/2 + TOP_Y - 2)
  # pyautogui.click(rando[0] + rando[2]/2 + TOP_X - 2, rando[1] + rando[3]/2 + TOP_Y + 2)


  # TESTING
  # detection_image = vision.draw_rectangles(screenshot_numpy, rectangles)
  # cv.imshow('look at all those chickens', detection_image)
  # key = cv.waitKey(1)
  # if key == ord('q'):
  #   cv.destroyAllWindows()
  #   break
