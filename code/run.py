import cv2
import Checker

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 0
params.maxThreshold = 240
detector = cv2.SimpleBlobDetector_create(params)

C = CheckerChecker(detector,path=IMG)
C.analyze_board()
