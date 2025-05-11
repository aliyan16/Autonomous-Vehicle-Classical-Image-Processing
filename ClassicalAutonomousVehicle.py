import numpy as np
# pandas and matplotlib.pyplot are not used in your core functions,
# you can comment them out if not needed for other developments.
# import pandas as pd
# import matplotlib.pyplot as plt
import cv2 as cv

# --- [Your existing functions: smoothing, cannyEdge, remove_noise, preprocess_image, region_of_interest, increase_contrast, hough_transform, average_slope_intercept, pixel_points, get_lane_lines, draw_lines_on_image, draw_lane_area] ---
# (All your functions from the previous script should be here)

def smoothing(img):
    kernel=5
    smoothImg=cv.GaussianBlur(img,(kernel, kernel), 0)
    return smoothImg

def cannyEdge(img, low_threshold=80, high_threshold=200): # Added parameters for flexibility
    return cv.Canny(img, low_threshold, high_threshold)

def remove_noise(edges):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opened = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel)
    return opened

def preprocess_image(img):
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced

def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height), (width, height),
        (int(width*0.55), int(height*0.55)), (int(width*0.45), int(height*0.55))
    ]], dtype=np.int32)
    cv.fillPoly(mask, polygon, 255)
    return cv.bitwise_and(img, mask)

def increase_contrast(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(12,12))
        cl = clahe.apply(l)
        blended_l = cv.addWeighted(l, 0.3, cl, 0.7, 0)
        limg = cv.merge((blended_l,a,b))
        return cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        gray_img = img.copy()
        if len(img.shape) == 3 and img.shape[2] == 1: gray_img = img[:,:,0]
        clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(12,12))
        enhanced = clahe.apply(gray_img)
        return cv.addWeighted(gray_img, 0.3, enhanced, 0.7, 0)
    else:
        return img

def hough_transform(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=30):
    return cv.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    
def average_slope_intercept(lines, image_width, image_height):
    left_lines, left_weights = [], []
    right_lines, right_weights = [], []
    min_abs_slope, max_abs_slope = 0.25, 2.0

    if lines is None: return None, None
    for segment in lines:
        for x1, y1, x2, y2 in segment:
            if x1 == x2: continue
            slope = (y2 - y1) / (x2 - x1)
            if not (min_abs_slope < abs(slope) < max_abs_slope): continue
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            x_at_bottom = (image_height - intercept) / slope
            if slope < 0: # Left lane
                if 0 < x_at_bottom < image_width * 0.5: # Consider tuning 0.5 to 0.45-0.48
                    left_lines.append((slope, intercept)); left_weights.append(length)
            else: # Right lane
                if image_width * 0.5 < x_at_bottom < image_width: # Consider tuning 0.5 to 0.52-0.55
                    right_lines.append((slope, intercept)); right_weights.append(length)
    left_lane  = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
    return left_lane, right_lane
    
def pixel_points(y1, y2, line):
    if line is None: return None
    slope, intercept = line
    if abs(slope) < 1e-4: return None
    x1 = int((y1 - intercept) / slope); x2 = int((y2 - intercept) / slope)
    return ((x1, int(y1)), (x2, int(y2)))
    
def get_lane_lines(image, hough_lines):
    h, w = image.shape[:2]
    left_params, right_params = average_slope_intercept(hough_lines, w, h)
    y1, y2 = h, h * 0.6 # Lines from bottom to 60% of height
    return pixel_points(y1, y2, left_params), pixel_points(y1, y2, right_params)
    
def draw_lines_on_image(image, lines_pts, color=[255, 0, 0], thickness=10): # BGR: Blue
    output_image = image.copy() # Work on a copy
    line_mask = np.zeros_like(output_image)
    for line_pt in lines_pts:
        if line_pt: cv.line(line_mask, *line_pt, color, thickness)
    # Add the lines to the image. If 'output_image' already has filled area, lines are on top.
    return cv.addWeighted(output_image, 1.0, line_mask, 1.0, 0.0)

def draw_lane_area(image, left_pts, right_pts, color=[0, 255, 0], alpha=0.3): # BGR: Green
    output_image = image.copy()
    if not (left_pts and right_pts): return output_image
    overlay = output_image.copy()
    pts = np.array([left_pts[0], left_pts[1], right_pts[1], right_pts[0]], dtype=np.int32)
    cv.fillPoly(overlay, [pts], color)
    cv.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, dst=output_image)
    return output_image
# ---------------------------------------------------------------------------

if __name__=='__main__':
    videoPath = r'C:\AllData\Selfskills\MachineLearning&ComputerVision\DipProject\Autonomous-Vehicle-Classical-Image-Processing\DIP Project Videos\PXL_20250325_045117252.TS.mp4'
    cap = cv.VideoCapture(videoPath)

    if not cap.isOpened():
        print(f"Error: Could not open video file {videoPath}")
        exit()

    processing_width, processing_height = 512, 512

    # Variables for simple temporal smoothing
    last_known_left_lane_points = None
    last_known_right_lane_points = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video or error in reading frame.")
            break

        frame_resized = cv.resize(frame, (processing_width, processing_height))
        processed_frame_output = frame_resized.copy() # This will be the frame we draw final results on

        # --- Processing Pipeline ---
        high_contrast_img = increase_contrast(frame_resized)
        enhanced_gray = preprocess_image(high_contrast_img)
        median_blurred = cv.medianBlur(enhanced_gray, 5)
        gaussian_blurred = smoothing(median_blurred)
        
        # Canny Edge Detection - Experiment with thresholds here!
        canny_img = cannyEdge(gaussian_blurred, low_threshold=60, high_threshold=180) # Example: Adjusted thresholds
        
        roc_img = region_of_interest(canny_img)
        
        # Hough Transform - Experiment with parameters here!
        hough_lines = hough_transform(roc_img, threshold=20, minLineLength=20, maxLineGap=40) # Example: Adjusted parameters

        # Get current frame's lane line coordinates
        current_left_points, current_right_points = get_lane_lines(frame_resized, hough_lines)

        # --- Temporal Smoothing ---
        if current_left_points is not None:
            last_known_left_lane_points = current_left_points
        else:
            # If current detection failed for left, use the last known good one (if any)
            current_left_points = last_known_left_lane_points 

        if current_right_points is not None:
            last_known_right_lane_points = current_right_points
        else:
            # If current detection failed for right, use the last known good one (if any)
            current_right_points = last_known_right_lane_points
        
        # --- Debug Visualizations (Uncomment to use when detection fails) ---
        # if current_left_points is None or current_right_points is None: # Or based on some other failure criteria
        # print("Debug: Detection issue on this frame.")
        # cv.imshow('Debug - Resized Frame', frame_resized)
        # cv.imshow('Debug - Canny Edges', canny_img)
        # cv.imshow('Debug - ROI Edges', roc_img)
        #     if hough_lines is not None:
        #         debug_hough_on_roi = cv.cvtColor(roc_img.copy(), cv.COLOR_GRAY2BGR)
        # for line_segment in hough_lines:
        # for x1,y1,x2,y2 in line_segment:
        #                 cv.line(debug_hough_on_roi,(x1,y1),(x2,y2),(0,255,255),1) # Yellow lines
        #         cv.imshow('Debug - Raw Hough Lines on ROI', debug_hough_on_roi)
        #     cv.waitKey(0) # Pause to inspect

        # --- Drawing ---
        # 1. Draw Lane Area (Filled Polygon) using (potentially smoothed) points
        if current_left_points and current_right_points:
            processed_frame_output = draw_lane_area(processed_frame_output, current_left_points, current_right_points)

        # 2. Draw Lane Lines using (potentially smoothed) points
        lines_to_draw = []
        if current_left_points: lines_to_draw.append(current_left_points)
        if current_right_points: lines_to_draw.append(current_right_points)
        
        if lines_to_draw:
            processed_frame_output = draw_lines_on_image(processed_frame_output, lines_to_draw)

        cv.imshow('Lane Detection Video', processed_frame_output)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()