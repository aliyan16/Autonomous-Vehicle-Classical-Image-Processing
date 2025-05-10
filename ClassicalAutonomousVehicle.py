import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv



def smoothing(img):
    kernel=5
    smoothImg=cv.GaussianBlur(img,(kernel, kernel), 0)
    return smoothImg

def cannyEdge(img):


    return cv.Canny(img,50,150)

def preprocess_image(img):
    """Convert to grayscale and enhance contrast"""
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    return enhanced
def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width*0.55), int(height*0.55)),
        (int(width*0.45), int(height*0.55))
    ]], dtype=np.int32)
    cv.fillPoly(mask, polygon, 255)
    return cv.bitwise_and(img, mask)


if __name__=='__main__':
    imgPath = r'C:\AllData\Selfskills\MachineLearning&ComputerVision\DipProject\Autonomous-Vehicle-Classical-Image-Processing\Images\frame1.jpg'
    img = cv.imread(imgPath)
    img=cv.resize(img,(512,512))
    cv.imshow('original img',img)
    cv.waitKey(2000)
    enhanced = preprocess_image(img)
    cv.imshow('preprocessed img',enhanced)
    cv.waitKey(2000)
    gaussianImg=smoothing(enhanced)
    cannyImg=cannyEdge(gaussianImg)
    # cannyImg=cv.resize(cannyImg,(512,512))
    cv.imshow('edge img',cannyImg)
    cv.waitKey(2000)
    rocImg=region_of_interest(cannyImg)
    cv.imshow('roc img',rocImg)
    cv.waitKey(0)
