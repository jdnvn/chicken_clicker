import time
from pynput import mouse, keyboard
from apscheduler.schedulers.background import BackgroundScheduler
import threading
import numpy as np
from PIL import ImageGrab
import cv2

TOP_X = 4
TOP_Y = 51
BOTTOM_X = 600
BOTTOM_Y = 430

def take_screenshot(x = 0.0, y = 0.0):
  screenshot = ImageGrab.grab(bbox=(TOP_X, TOP_Y, BOTTOM_X, BOTTOM_Y))
  screenshot_numpy = np.array(screenshot, dtype='uint8'))


def start_scheduler():
  scheduler = BackgroundScheduler()
  scheduler.add_job(take_screenshot, 'interval', seconds=3)

  try:
    scheduler.start()

    while True:
      time.sleep(1)
  except (KeyboardInterrupt, SystemExit):
     scheduler.shutdown()

def main():
  try:
    scheduler_thread = threading.Thread(target=start_scheduler)
    scheduler_thread.start()

    def on_click(x, y, button, pressed):
      if pressed and x >= TOP_X and x <= BOTTOM_X and y >= TOP_Y and y <= BOTTOM_Y:
        x_adj = x - TOP_X
        y_adj = y - TOP_Y
        take_screenshot(x=x_adj, y=y_adj)

    def on_press(key):
        try:
            print(f'{key}')
        except AttributeError:
            print(f'{key}')


    # Collect mouse events
    mouse_listener = mouse.Listener(on_click=on_click)
    mouse_listener.start()

    # # Collect keyboard events
    # keyboard_listener = keyboard.Listener(on_press=on_press)
    # keyboard_listener.start()

    # # Keep the script running
    # keyboard_listener.join()
    mouse_listener.join()
  finally:
    print("closing...")
    np.save('coords_array', coords_array)
    np.save('image_array', image_array)

if __name__=='__main__':
    main()


