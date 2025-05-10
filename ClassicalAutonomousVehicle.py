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
   

def hough_transform(image):
    """
    Determine and cut the region of interest in the input image.
    Parameter:
        image: grayscale image which should be an output from the edge detector
    """
    # Distance resolution of the accumulator in pixels.
    rho = 1             
    # Angle resolution of the accumulator in radians.
    theta = np.pi/180   
    # Only lines that are greater than threshold will be returned.
    threshold = 20      
    # Line segments shorter than that are rejected.
    minLineLength = 20  
    # Maximum allowed gap between points on the same line to link them
    maxLineGap = 500    
    # function returns an array containing dimensions of straight lines 
    # appearing in the input image
    return cv.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,minLineLength = minLineLength, maxLineGap = maxLineGap)
   
def average_slope_intercept(lines):
    """
    Find the slope and intercept of the left and right lanes of each image.
    Parameters:
        lines: output from Hough Transform
    """
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
     
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # slope of left lane is negative and for right lane slope is positive
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    # 
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane
   
def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))
   
def lane_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line
 
     
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image (video frame in our case).
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 12): Line thickness. 
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv.line(line_image, *line,  color, thickness)
    return cv.addWeighted(image, 1.0, line_image, 1.0, 0.0)

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
    cv.waitKey(2000)
    HoughImg=hough_transform(rocImg)
    result = draw_lane_lines(img, lane_lines(img, HoughImg))
    cv.imshow('Lines',result)
    cv.waitKey(0)
