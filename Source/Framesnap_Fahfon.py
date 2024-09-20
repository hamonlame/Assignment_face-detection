import cv2
import os
import time

# Open the default camera
cam = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

# Load Haarcascade for face detection
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if Haarcascade is loaded successfully
if haar_cascade.empty():
    print("Error: Could not load Haarcascade XML file.")
    exit()

# Create directories to save images
output_dir = 'Output/Fahfon_UXUI'
os.makedirs(output_dir, exist_ok=True)

frame_count = 0  # To keep track of the number of saved images
last_save_time = time.time()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply face detection on the grayscale frame
    faces_rect = haar_cascade.detectMultiScale(gray_frame, 1.1, 9)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces_rect:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the frame if a face is detected and within the limit
        current_time = time.time()
        if frame_count < 100 and (current_time - last_save_time) >= 5:
            frame_count += 1
            image_path = os.path.join(output_dir, f'{frame_count}_Fahfon.jpg')
            cv2.imwrite(image_path, frame)
            print(f"Saved: {image_path}")
            last_save_time = current_time

    # Write the frame to the output file
    out.write(frame)

    # Display the frame with face detection
    cv2.imshow('Camera - Face Detection', frame)

    # Check for key press to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Exiting...")
        break

# Release the capture and writer objects
cam.release()
out.release()
print("Camera and VideoWriter released.")
cv2.destroyAllWindows()
