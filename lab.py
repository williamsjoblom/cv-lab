import cv2

import vision

def find_ball(image):
    colorLower = (3, 100, 2)
    colorUpper = (35, 255, 255)
    
    image.blur(2)
    image.in_range(colorLower, colorUpper)

    image.erode(2)
    image.dilate(2)
    image.show()

    
vision.run(find_ball)
