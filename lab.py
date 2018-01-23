import cv2

import sys, lab


def find_ball(image):
    colorLower = (3, 100, 2)
    colorUpper = (35, 255, 255)
    
    image.blur(2)
    image.in_range(colorLower, colorUpper)

    image.erode(2)
    image.dilate(2)
    image.show()

    
lab.run(find_ball)
