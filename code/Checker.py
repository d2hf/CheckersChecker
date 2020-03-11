import numpy as np
import cv2
from time import sleep
import matplotlib.pyplot as plt

class CheckersChecker():
    
    '''
    An object that analyzes a checker board image and provides
    data about the image provided.
    
    The checkers colors must be green and red.
    '''
    
    def __init__(self,detector,path):
        self.img_path = path
        self.detector = detector
        
        self.bgr = self._read_img()
        self.hsv = self._convert_hsv()
        self.rgb = self._convert_rgb()
    
    
    def _read_img(self):
        return cv2.imread(self.img_path)
    
    
    def _convert_hsv(self):
        return cv2.cvtColor(self.bgr,cv2.COLOR_BGR2HSV)
    
    
    def _convert_rgb(self):
        return cv2.cvtColor(self.bgr,cv2.COLOR_BGR2RGB)
    
    
    def _gen_red_mask(self):
        lowerLimit = np.array([0, 70, 50])
        upperLimit = np.array([10, 255, 255])

        mask1 = cv2.inRange(self.hsv, lowerLimit, upperLimit)

        lowerLimit = np.array([170, 70, 50])
        upperLimit = np.array([180, 255, 255])

        mask2 = cv2.inRange(self.hsv, lowerLimit, upperLimit)
        
        return mask1 + mask2
    
    
    def _gen_green_mask(self):
        lowerLimit = np.array([40, 70, 70])
        upperLimit = np.array([80, 255, 255])
        
        mask = cv2.inRange(self.hsv, lowerLimit, upperLimit)
        
        return mask
    
    
    def count_red(self):
        mask = self._gen_red_mask()
        
        res = cv2.bitwise_and(self.hsv,self.hsv, mask= mask)
        blur = cv2.GaussianBlur(res,(3,3),0)
        gray = blur[:,:,2]
        self.red_img = cv2.bitwise_not(gray)
        keyp = self.detector.detect(self.red_img)
        
        return len(keyp)
    
    
    def count_green(self):
        mask = self._gen_green_mask()
        
        res = cv2.bitwise_and(self.hsv,self.hsv, mask= mask)
        blur = cv2.GaussianBlur(res,(3,3),0)
        gray = blur[:,:,2]
        self.green_img = cv2.bitwise_not(gray)
        keyp = self.detector.detect(self.green_img)
        
        return len(keyp)
    
    def analyze_board(self):
        print("BOARD ANALYSIS:\n# of greens: {}\n# of reds: {}".format(self.count_green(),
                                                                      self.count_red()))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15,5))
        
        ax1.set_title('original board')
        ax1.imshow(self.rgb)
        
        ax2.set_title('green checkers')
        ax2.imshow(self.green_img, cmap='gray')
        
        ax3.set_title('red checkers')
        ax3.imshow(self.red_img, cmap='gray')
        
