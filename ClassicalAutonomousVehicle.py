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
   


def hough_transform(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=30): # Adjusted maxLineGap
    """
    Determine and cut the region of interest in the input image.
    Parameter:
        image: grayscale image which should be an output from the edge detector
    """
    return cv.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    
def average_slope_intercept(lines, image_width, image_height):
    """
    Find the slope and intercept of the left and right lanes of each image.
    Filters lines based on slope and position.
    Parameters:
        lines: output from Hough Transform
        image_width: width of the image for positional filtering
        image_height: height of the image for positional filtering
    """
    left_lines    = [] #(slope, intercept)
    left_weights  = [] #(length,)
    right_lines   = [] #(slope, intercept)
    right_weights = [] #(length,)
    
    min_abs_slope = 0.25 # Avoids very flat lines (tune this)
    max_abs_slope = 2.0  # Avoids very steep lines (tune this)

    if lines is None:
        return None, None

    for line_segment in lines:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2: # vertical line
                continue 
            
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))

            if not (min_abs_slope < abs(slope) < max_abs_slope):
                continue

            # Calculate x-coordinate where the line would intersect the bottom of the image
            # This helps filter lines that are not in the correct half of the road
            x_at_bottom = (image_height - intercept) / slope
            
            if slope < 0: # Left lane
                # Expect x_at_bottom to be in the left part of the image, e.g., less than center
                # Further refinement: x_at_bottom should not be too far left either (e.g. > image_width * 0.1)
                if 0 < x_at_bottom < image_width * 0.5: # Tune 0.5 (e.g., to 0.48 or based on ROI)
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
            else: # Right lane
                # Expect x_at_bottom to be in the right part of the image
                # Further refinement: x_at_bottom should not be too far right (e.g. < image_width * 0.9)
                if image_width * 0.5 < x_at_bottom < image_width: # Tune 0.5 (e.g., to 0.52 or based on ROI)
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
                
    left_lane  = np.dot(left_weights,  left_lines) / np.sum(left_weights)  if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    
    return left_lane, right_lane
    
def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
    """
    if line is None:
        return None
    slope, intercept = line
    if abs(slope) < 1e-4 : # Avoid division by zero or very large x for near-horizontal lines
        return None
        
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))
    
def get_lane_lines(image, hough_lines): # Renamed to avoid conflict, more descriptive
    """
    Create full length lines from pixel points.
    Parameters:
        image: The original image (used for dimensions).
        hough_lines: The output lines from Hough Transform.
    """
    img_height, img_width = image.shape[:2]
    
    left_lane_params, right_lane_params = average_slope_intercept(hough_lines, img_width, img_height)
    
    # Define y-coordinates for drawing the final lines
    y1 = img_height       # bottom of the image
    y2 = img_height * 0.6 # Top of the detected lane line (can be float, pixel_points will int it)
    # Alternative: y2 = int(img_height * 0.55) # To align with ROI top

    left_line_pts  = pixel_points(y1, y2, left_lane_params)
    right_line_pts = pixel_points(y1, y2, right_lane_params)
    
    return left_line_pts, right_line_pts
    
def draw_lines_on_image(image, lines_pts, color=[0, 0, 255], thickness=10): # Renamed
    """
    Draw lines onto the input image.
    Parameters:
        image: The input test image.
        lines_pts: A list of line coordinates (e.g., [left_line_pts, right_line_pts]).
        color: Line color.
        thickness: Line thickness. 
    """
    line_image = np.zeros_like(image)
    for line_pt in lines_pts:
        if line_pt is not None:
            cv.line(line_image, *line_pt,  color, thickness)
    # Blend lines with original image
    return cv.addWeighted(image, 0.8, line_image, 1.0, 0.0) # Original image weight 1.0 in user code

def draw_lane_area(image, left_line_pts, right_line_pts, color=[0, 255, 0], alpha=0.3):
    """
    Fills the area between the two detected lane lines with a semi-transparent color.
    """
    if left_line_pts is None or right_line_pts is None:
        return image

    overlay = image.copy()
    
    # Define the polygon points for the lane area
    # Order: bottom-left, top-left, top-right, bottom-right
    pts = np.array([left_line_pts[0], left_line_pts[1], right_line_pts[1], right_line_pts[0]], dtype=np.int32)
    
    cv.fillPoly(overlay, [pts], color)
    
    # Blend the filled area with the original image
    return cv.addWeighted(overlay, alpha, image, 1 - alpha, 0)




if __name__=='__main__':
    imgPath = r'C:\AllData\Selfskills\MachineLearning&ComputerVision\DipProject\Autonomous-Vehicle-Classical-Image-Processing\Images\frame1.jpg'
    img = cv.imread(imgPath)
    img=cv.resize(img,(512,512))
    cv.imshow('original img',img)
    cv.waitKey(1000)
    high_contrast_img = increase_contrast(img)
    cv.imshow('high contrast img', high_contrast_img)
    cv.waitKey(1000)
    enhanced = preprocess_image(high_contrast_img)
    cv.imshow('preprocessed img',enhanced)
    cv.waitKey(1000)
    medianBlur=cv.medianBlur(enhanced,5)
    gaussianImg=smoothing(medianBlur)
    cannyImg=cannyEdge(gaussianImg)
    # CleancannyImg=remove_noise(cannyImg)
    # cannyImg=cv.resize(cannyImg,(512,512))
    cv.imshow('edge img',cannyImg)
    cv.waitKey(1000)
    rocImg=region_of_interest(cannyImg)
    cv.imshow('roc img',rocImg)
    cv.waitKey(1000)
    HoughImg=hough_transform(rocImg)
    left_lane_points,right_lane_points=get_lane_lines(img, HoughImg)
    result = draw_lines_on_image(img,(left_lane_points,right_lane_points))
    cv.imshow('Lines',result)
    cv.waitKey(1000)
    output_image = draw_lane_area(result, left_lane_points, right_lane_points, color=[0, 255, 0], alpha=0.3) # Green fill
    cv.imshow('Final image',output_image)
    cv.waitKey(0)

