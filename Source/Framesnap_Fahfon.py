import cv2
import os
import time

# Function to find the next folder number
def get_next_folder_number(output_dir):
    if not os.path.exists(output_dir):
        return 1
    folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    person_folders = [int(f.split('_')[1]) for f in folders if f.startswith('Person_') and f.split('_')[1].isdigit()]
    return max(person_folders) + 1 if person_folders else 1

# Open the default camera
cam = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load Haarcascade for face detection
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if Haarcascade is loaded successfully
if haar_cascade.empty():
    print("Error: Could not load Haarcascade XML file.")
    exit()

# Initial settings for folder and image count
output_dir = 'Output'
person_count = get_next_folder_number(output_dir)  # Get the next available folder number
image_count = 0  # To keep track of images in the current folder
max_images_per_folder = 10  # Maximum images per folder
last_save_time = time.time()

# Create the first folder
current_folder = f'{output_dir}/Person_{person_count}'
os.makedirs(current_folder, exist_ok=True)

# Reset global image number for the first folder
global_image_number = 1

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally to act like a mirror
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply face detection on the grayscale frame
    faces_rect = haar_cascade.detectMultiScale(gray_frame, 1.1, 9)

    # Draw rectangles around detected faces and save images
    for (x, y, w, h) in faces_rect:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the frame if a face is detected and 2 seconds have passed since the last save
        current_time = time.time()
        if image_count < max_images_per_folder and (current_time - last_save_time) >= 2:
            image_count += 1
            # Save the image with global image number
            image_path = os.path.join(current_folder, f'{global_image_number}.jpg')
            cv2.imwrite(image_path, frame)
            print(f"Saved: {image_path}")
            last_save_time = current_time
            global_image_number += 1  # Increment the global image number

        # If 10 images have been saved, show a countdown before creating a new folder
        if image_count >= max_images_per_folder:
            countdown_start_time = time.time()  # Record the start time of the countdown
            countdown_duration = 10  # Countdown from 10 seconds

            while time.time() - countdown_start_time < countdown_duration:
                ret, frame = cam.read()
                if not ret:
                    break
                # Flip the frame
                frame = cv2.flip(frame, 1)

                # Apply face detection on the grayscale frame
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_rect = haar_cascade.detectMultiScale(gray_frame, 1.1, 9)
                for (x, y, w, h) in faces_rect:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate the remaining time for the countdown
                countdown_time_left = countdown_duration - int(time.time() - countdown_start_time)

                # Set font scale and thickness for large text
                font_scale = 5
                thickness = 5

                # Get the size of the text to center it
                text = str(countdown_time_left)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_x = int((frame_width - text_size[0]) / 2)
                text_y = int((frame_height + text_size[1]) / 2)

                # Display the countdown number in large font at the center of the frame
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

                # Show the countdown frame with face detection
                cv2.imshow('Camera - Face Detection', frame)

                # Reduce wait time to 1 millisecond for smoother updates
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    cam.release()
                    cv2.destroyAllWindows()
                    exit()  # Exit the entire program

            # Create a new folder for the next person
            person_count += 1
            image_count = 0  # Reset image count for the new folder
            current_folder = f'{output_dir}/Person_{person_count}'
            os.makedirs(current_folder, exist_ok=True)
            print(f"Created new folder: {current_folder}")

            # Reset global image number for the new folder
            global_image_number = 1

    # Display the frame with face detection
    cv2.imshow('Camera - Face Detection', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the capture object
cam.release()
print("Camera released.")
cv2.destroyAllWindows()
