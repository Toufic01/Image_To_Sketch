# Importing library
import os
import cv2
import numpy as np

# Create folders to store output images if they don't exist
output_folder = "output_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Convert to pencil sketch
def pencil_sketch(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to gray scale

    # Invert the gray image
    inverted_gray = cv2.bitwise_not(gray_image)

    # Apply Gaussian blur to the inverted image
    blurred_image = cv2.GaussianBlur(inverted_gray, (21, 21), 0)

    # Invert the blurred image
    inverted_blurred = cv2.bitwise_not(blurred_image)

    # Create the pencil sketch by dividing the gray image by the inverted blurred image
    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    return pencil_sketch

# Converting to cartoon image
def cartoonize(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to gray scale
    gray_image = cv2.medianBlur(gray_image, 5) # applying median blur to get cartoon image
    edges = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color_image = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon_image = cv2.bitwise_and(color_image, color_image, mask=edges)

    return cartoon_image

# Main function
def main():
    cap = cv2.VideoCapture(0)
    capturing = False

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if capturing:
            sketch = pencil_sketch(frame)
            cartoon = cartoonize(frame)

            # Apply visual effects to the camera panel
           # frame = cv2.flip(frame, 1) # Flip the frame horizontally
           # frame = cv2.putText(frame, "HACKING IN PROGRESS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Camera Frame", frame)

            # Apply visual effects to the output panel
            sketch = cv2.putText(sketch, "Pencil Sketch", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cartoon = cv2.putText(cartoon, "Cartoon Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Pencil Sketch", sketch)
            cv2.imshow("Cartoon Image", cartoon)

            # Save the images in JPG format
            sketch_filename = os.path.join(output_folder, "sketch_image.jpg")
            cartoon_filename = os.path.join(output_folder, "cartoon_image.jpg")
            cv2.imwrite(sketch_filename, sketch, [cv2.IMWRITE_JPEG_QUALITY, 100])
            cv2.imwrite(cartoon_filename, cartoon, [cv2.IMWRITE_JPEG_QUALITY, 100])

            print(f"Sketch saved as {sketch_filename}")
            print(f"Cartoon image saved as {cartoon_filename}")

            capturing = False
        else:
            # Apply visual effects to the camera panel
           # frame = cv2.flip(frame, 1) # Flip the frame horizontally
           # frame = cv2.putText(frame, "HACKING IN PROGRESS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
          #  frame = np.random.randint(0, 256, frame.shape).astype(np.uint8) # Add random noise

            cv2.imshow("Camera Frame", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('c'):
            capturing = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
