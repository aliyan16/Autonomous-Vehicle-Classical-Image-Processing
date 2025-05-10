import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv



def smoothing(img):
    kernel=5
    smoothImg=cv.GaussianBlur(img,(kernel, kernel), 0)
    return smoothImg

def cannyEdge(img):


    return cv.Canny(img,80,200)
def remove_noise(edges):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opened = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel)
    return opened


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
def increase_contrast(img):
   """Best balance for lane detection"""
   if len(img.shape) == 3:
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
        # Moderate CLAHE settings
    clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(12,12))
    cl = clahe.apply(l)
        # Blend 70% of enhanced with 30% original
    blended_l = cv.addWeighted(l, 0.3, cl, 0.7, 0)
    limg = cv.merge((blended_l,a,b))
    return cv.cvtColor(limg, cv.COLOR_LAB2BGR)
   else:
    clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(12,12))
    enhanced = clahe.apply(img)
    return cv.addWeighted(img, 0.3, enhanced, 0.7, 0)
   


if __name__=='__main__':
    imgPath = r'C:\AllData\Selfskills\MachineLearning&ComputerVision\DipProject\Autonomous-Vehicle-Classical-Image-Processing\Images\frame1.jpg'
    img = cv.imread(imgPath)
    img=cv.resize(img,(512,512))
    cv.imshow('original img',img)
    cv.waitKey(2000)
    high_contrast_img = increase_contrast(img)
    cv.imshow('high contrast img', high_contrast_img)
    cv.waitKey(2000)
    enhanced = preprocess_image(high_contrast_img)
    cv.imshow('preprocessed img',enhanced)
    cv.waitKey(2000)
    medianBlur=cv.medianBlur(enhanced,5)
    gaussianImg=smoothing(medianBlur)
    cannyImg=cannyEdge(gaussianImg)
    # CleancannyImg=remove_noise(cannyImg)
    # cannyImg=cv.resize(cannyImg,(512,512))
    cv.imshow('edge img',cannyImg)
    cv.waitKey(2000)
    rocImg=region_of_interest(cannyImg)
    cv.imshow('roc img',rocImg)
    cv.waitKey(0)
