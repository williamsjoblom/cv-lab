from collections import deque
import colorsys
import numpy as np
import argparse
import cv2

grayscale = False
BUF_SZ = 64
MIN_RADIUS = 40
        
class Image:
    def __init__(self, frame):
        self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.grayscale = False
        self.frame = frame
        self.pts = deque(maxlen=BUF_SZ)
    

    def show(self):
        cv2.imshow("Intermediate", self.hsv)
        
    def blur(self, n):
        self.hsv = cv2.GaussianBlur(self.hsv, (2*n - 1, 2*n - 1), 0)

    def erode(self, n):
        for i in xrange(n):
            self.hsv = cv2.erode(self.hsv, None, 2)

    def dilate(self, n):
        for i in xrange(n):
            self.hsv = cv2.dilate(self.hsv, None, 2)

    def in_range(self, lower, upper):
        self.grayscale = True
        if lower[0] > upper[0]: # Wrap around!
            delta_h = abs(255 - lower[0] + upper[0]) % 255
            delta_s = abs(upper[1] - lower[1]) % 255
            delta_v = abs(upper[2] - lower[2]) % 255

            wrap_amount = (float(255 - lower[0]) / delta_h)

            wrap_s = min(lower[1], upper[1]) + delta_s*(wrap_amount)
            wrap_v = min(lower[2], upper[2]) + delta_v*(wrap_amount)
            
            wrap_upper = np.array([255, int(wrap_s), int(wrap_v)], dtype=np.uint8)
            wrap_lower = np.array([0, int(wrap_s), int(wrap_v)], dtype=np.uint8)
            
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            mask0 = cv2.inRange(self.hsv, lower, wrap_upper)
            mask1 = cv2.inRange(self.hsv, wrap_lower, upper)

            self.hsv = mask0 + mask1
        else:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            self.hsv = cv2.inRange(self.hsv, lower, upper)

    def get_coord(self):
        if self.grayscale:
	    cnts = cv2.findContours(self.hsv, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]
        else:
            cnts = []
	    center = None
        x, y, radius = 0, 0, 0

        if len(cnts) > 0:
	    c = max(cnts, key=cv2.contourArea)
	    (x, y), radius = cv2.minEnclosingCircle(c)
	    if radius > MIN_RADIUS: 
                return x, y, radius
            else:
                print("Did not find any objects. No coordinates available")
                return 0,0,0
    def draw_circle(self, x, y, radius):
	cv2.circle(self.frame, (int(x), int(y)), int(radius),
		   (0, 255, 255), 2)
            

    
        
def run(callback):
    camera = cv2.VideoCapture(0)
    (grabbed, frame) = camera.read()
    cv2.namedWindow('Frame')
    image = Image(frame)
    while True:
        (grabbed, frame) = camera.read()
        image.frame = frame
        image.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        callback(image)
      	cv2.imshow("Frame", image.frame)
	key = cv2.waitKey(1) & 0xFF
        
	if key == ord("q"):
            break
    
    camera.release()
    cv2.destroyAllWindows()





    
"""
        if radius > MIN_RADIUS:
            pts.appendleft(center)

	for i in xrange(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
		continue

            thickness = int(np.sqrt(BUF_SZ / float(i + 1)) * 2.5)
	    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
	    




M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                
        if radius > MIN_RADIUS and (not center == None) :
            self.pts.appendleft(center)
                
	    for i in xrange(1, len(self.pts)):
                print("here")
                if self.pts[i - 1] is None or self.pts[i] is None:
		    continue
                
                thickness = int(np.sqrt(BUF_SZ / float(i + 1)) * 2.5)
	        cv2.line(self.frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)
                cv2.circle(self.frame, center, 5, (0, 0, 255), -1)
            

"""	    
