import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def conv(img, filter):
    img = img.astype(np.int16)
    rows, cols = img.shape
    value = 0
    for i in range(rows):
        for j in range(cols):
            value += img[i, j] * filter[i][j]
    return value


def norm(img):
    imgm = (img / np.max(img) * 255).astype(np.uint8)
    return imgm


def smoothing(img, filter):
    rows, cols = img.shape
    frows, fcols = filter.shape
    pad_h = frows // 2
    pad_w = fcols // 2
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    smoothImg = np.zeros_like(img)

    for i in range(rows):
        for j in range(cols):
            window = padded_img[i:i+frows, j:j+fcols]
            smoothImg[i, j] = conv(window, filter)
    return norm(smoothImg)


def sobel(img, sobelx, sobely):
    rows, cols = img.shape
    pad = 1
    padded_img = np.pad(img, ((pad, pad), (pad, pad)), mode='constant')

    gx = np.zeros_like(img)
    gy = np.zeros_like(img)

    for i in range(rows):
        for j in range(cols):
            window = padded_img[i:i+3, j:j+3]
            gx[i, j] = conv(window, sobelx)
            gy[i, j] = conv(window, sobely)
    return gx, gy


def magnitude(img1, img2):
    resultant = np.sqrt((img1 ** 2) + (img2 ** 2))
    return norm(resultant)


def Phase(img1, img2):
    resultant = np.arctan2(img2, img1) * (180.0 / np.pi)
    return norm(resultant)


def nonMaxSupress(gradient_magnitude, gradient_direction):
    suppressed = np.zeros_like(gradient_magnitude)
    angle = gradient_direction

    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            direction = angle[i, j]
            direction = ((direction + 22.5) // 45) * 45 % 180

            if direction == 0:
                neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
            elif direction == 45:
                neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]
            elif direction == 90:
                neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
            elif direction == 135:
                neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
            else:
                neighbors = [0, 0]

            if gradient_magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = gradient_magnitude[i, j]

    return suppressed


def cannyedge(img):
    gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobely = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    gaussianImg = smoothing(img, gaussian)
    gx, gy = sobel(gaussianImg, sobelx, sobely)
    SobelMag = magnitude(gx, gy)
    SobelPhase = Phase(gx, gy)
    nonmaxMagntitude = nonMaxSupress(SobelMag, SobelPhase)

    # Thresholding
    high_threshold = 150
    low_threshold = 50
    strong_edges = nonmaxMagntitude >= high_threshold
    weak_edges = (nonmaxMagntitude >= low_threshold) & (nonmaxMagntitude < high_threshold)
    edges = np.zeros_like(nonmaxMagntitude)
    edges[strong_edges] = 255

    return edges


def hough_transform(edge_image, theta_res=1, rho_res=1):
    height, width = edge_image.shape
    max_rho = int(np.ceil(np.sqrt(height**2 + width**2)))
    theta_values = np.deg2rad(np.arange(-90, 90, theta_res))
    accumulator = np.zeros((2 * max_rho, len(theta_values)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(edge_image)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for theta_idx, theta in enumerate(theta_values):
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = int(round(rho)) + max_rho
            accumulator[rho_idx, theta_idx] += 1

    return accumulator, theta_values, max_rho


def find_hough_peaks(accumulator, num_peaks=10, threshold=50, neighborhood_size=20):
    peaks = []
    acc_copy = accumulator.copy()

    for _ in range(num_peaks):
        max_idx = np.argmax(acc_copy)
        max_rho_idx, max_theta_idx = np.unravel_index(max_idx, acc_copy.shape)
        max_value = acc_copy[max_rho_idx, max_theta_idx]

        if max_value < threshold:
            break

        peaks.append((max_rho_idx, max_theta_idx, max_value))

        # Suppress neighborhood
        rho_min = max(0, max_rho_idx - neighborhood_size // 2)
        rho_max = min(acc_copy.shape[0], max_rho_idx + neighborhood_size // 2 + 1)
        theta_min = max(0, max_theta_idx - neighborhood_size // 2)
        theta_max = min(acc_copy.shape[1], max_theta_idx + neighborhood_size // 2 + 1)
        acc_copy[rho_min:rho_max, theta_min:theta_max] = 0

    return peaks


def draw_hough_lines(image, peaks, theta_values, max_rho):
    output_image = np.copy(image)
    if len(output_image.shape) == 2:
        output_image = np.stack((output_image,) * 3, axis=-1)

    for rho_idx, theta_idx, _ in peaks:
        theta = theta_values[theta_idx]
        rho = rho_idx - max_rho
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv.line(output_image, pt1, pt2, (255, 0, 0), 2)

    return output_image


# ====== MAIN EXECUTION ======
if __name__ == '__main__':
    img_path = r'C:\AllData\Selfskills\MachineLearning&ComputerVision\DipProject\Autonomous-Vehicle-Classical-Image-Processing\Images\frame1.jpg'
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (512, 512))  # resize to 512x512 for speed & clarity

    img_array = np.array(img)

    # Run Canny Edge Detection
    edges = cannyedge(img_array)

    # Run Hough Transform
    accumulator, theta_values, max_rho = hough_transform(edges)

    # Find peaks
    peaks = find_hough_peaks(accumulator, num_peaks=10)

    # Draw lines
    result = draw_hough_lines(img_array, peaks, theta_values, max_rho)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(img_array, cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.title("Edge Detection")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.title("Detected Lines")
    plt.imshow(result)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
