
import cv2

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize video capture object to capture video from webcam
vid_capture = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    _, image = vid_capture.read()
    
    # Convert the frame to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame using the cascade classifier
    faces_images = face_cascade.detectMultiScale(grayscale, 1.1, 4)
    
    # Draw a rectangle around each face
    for (xax, yax, wirdth, hiegth) in faces_images:
        cv2.rectangle(image, (xax, yax), (xax+wirdth, yax+hiegth), (255, 0, 0), 2)
    
    # Display the resulting image
    cv2.imshow('image', image)
    
    # Check if the user has pressed the 'Esc' key
   
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture object
vid_capture.release()
