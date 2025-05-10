import cv2
import os

# Set your paths
video_folder = r'C:\AllData\Selfskills\MachineLearning&ComputerVision\DipProject\Autonomous-Vehicle-Classical-Image-Processing\DIP Project Videos'
output_root = r'C:\AllData\Selfskills\MachineLearning&ComputerVision\DipProject\Autonomous-Vehicle-Classical-Image-Processing\ExtractedFrames'

# Create output directory if it doesn't exist
os.makedirs(output_root, exist_ok=True)

# Process all video files in the folder
for video_file in os.listdir(video_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add other video formats if needed
        video_path = os.path.join(video_folder, video_file)
        
        # Create a subfolder for this video's frames
        video_name = os.path.splitext(video_file)[0]
        output_folder = os.path.join(output_root, video_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Save every nth frame (adjust as needed)
            if frame_count % 5 == 0:  # Change this to 1 to save every frame
                frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
                cv2.imwrite(frame_path, frame)
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {frame_count} frames from {video_file} to {output_folder}")