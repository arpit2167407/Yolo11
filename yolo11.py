# Detection of First Image
from ultralytics import YOLO
import cv2
model=YOLO("yolo11n.pt")
model.to("mps")
results=model("/Users/arpitsharma/cvision/Images/363140724.jpg")
results[0].show()

#Detection of Multiple Images
import os
image_directory = "/Users/arpitsharma/cvision/Images"

# Get all image file paths
all_images = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Sort the images alphabetically (or modify sorting criteria as needed)
all_images = sorted(all_images)

# Select the first 5 images
first_5_images = all_images[:5]

# Process each image
for image_path in first_5_images:
    print(f"Processing: {image_path}")
    results = model(image_path)
    for result in results:
        result.show()  

#Detection of a video
x=model("/Users/arpitsharma/cvision/elephants.mp4",save=True)

# Detection of Objects in real time
model(0,save=False)

# Detection of a object in real time if camera is not opened automatically.
# Open webcam
cap = cv2.VideoCapture(0)  # Ensure the correct index

if not cap.isOpened():
    print("Cannot open the webcam!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Predict on the frame
    results = model(frame)

    # Display results
    annotated_frame = results[0].plot()  # Get annotated frame
    cv2.imshow("Webcam", annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
