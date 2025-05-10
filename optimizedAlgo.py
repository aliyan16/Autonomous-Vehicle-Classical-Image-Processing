import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def region_of_interest(img, vertices=None):
    """
    Applies an image mask to focus only on the region defined by the polygon
    formed from vertices. The rest of the image is set to black.
    """
    height, width = img.shape
    mask = np.zeros_like(img)
    
    # Define a default trapezoidal region if vertices not provided
    if vertices is None:
        bottom_left = (width * 0.1, height)
        top_left = (width * 0.4, height * 0.6)
        top_right = (width * 0.6, height * 0.6)
        bottom_right = (width * 0.9, height)
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    
    # Fill the polygon with white
    cv.fillPoly(mask, vertices, 255)
    
    # Apply the mask
    masked_image = cv.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=(0, 0, 255), thickness=5):
    """
    Draws lines on the image with optional color and thickness.
    """
    line_img = np.zeros_like(img) if len(img.shape) == 2 else np.copy(img)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(line_img, (x1, y1), (x2, y2), color, thickness)
    
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    Blend two images together according to weights α and β.
    """
    return cv.addWeighted(initial_img, α, img, β, γ)

def process_image(image_path):
    # Read and preprocess image
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    img = cv.resize(img, (640, 480))  # More standard size
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv.Canny(blur, threshold1=50, threshold2=150, apertureSize=3)
    
    # Apply region of interest mask (trapezoidal shape)
    roi_edges = region_of_interest(edges)
    
    # Detect lines using Probabilistic Hough Transform
    lines = cv.HoughLinesP(
        roi_edges, 
        rho=2, 
        theta=np.pi/180, 
        threshold=50, 
        minLineLength=50, 
        maxLineGap=30
    )
    
    # Create a blank image to draw lines on
    line_img = np.zeros_like(img)
    
    # Draw all detected lines in green
    line_img = draw_lines(line_img, lines, color=(0, 255, 0), thickness=5)
    
    # Combine the line image with the original
    result = weighted_img(line_img, img)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(132)
    plt.title("Canny Edges + ROI")
    plt.imshow(roi_edges, cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.title("Detected Lane Lines")
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    img_path = r'C:\AllData\Selfskills\MachineLearning&ComputerVision\DipProject\Autonomous-Vehicle-Classical-Image-Processing\Images\frame1.jpg'
    process_image(img_path)