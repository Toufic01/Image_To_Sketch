# Importing library
import os # System library
import cv2


# Create folders to store output images if they don't exist

output_folder = "output_images"

if not os.path.exists(output_folder):

    os.makedirs(output_folder)

# Convert to pencil sketch
def pencil_sketch(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to gray scale

    inverted_gray = 255 - gray_image # Adjust the light

    blurred_image = cv2.GaussianBlur(inverted_gray, (71, 71), 0) # filtering the image

    inverted_blurred = 255 - blurred_image # refiltering

    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0) # marging to get right sketch

    return pencil_sketch


# converting to cartoon image
def cartoonize(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to gray scale

    gray_image = cv2.medianBlur(gray_image, 5) # appling medianblur to get cartoon image

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
