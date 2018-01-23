import cv2

import vision

def find_ball(image):
    colorLower = (3, 100, 2)
    colorUpper = (35, 255, 255)
    
    image.blur(100)
    #image.in_range(colorLower, colorUpper)
    image.show()

    
vision.run(find_ball)
