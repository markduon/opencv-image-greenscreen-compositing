# Author: Quang Vinh Duong
# Date: 20/09/2024


import cv2
import numpy as np
import sys


# Utility functions for both tasks
def resize_image(image, target_width):
    # Calculate the aspect ratio and resize image while maintaining the aspect ratio
    aspect_ratio = image.shape[0] / image.shape[1]
    new_height = int(target_width * aspect_ratio)
    resized_image = cv2.resize(image, (target_width, new_height))
    return resized_image

def convert_and_split_image(image, color_space):
    if color_space == '-XYZ':
        converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
        c1, c2, c3 = cv2.split(converted_img)
    elif color_space == '-Lab':
        converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        c1, c2, c3 = cv2.split(converted_img)
    elif color_space == '-YCrCb':
        converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        c1, c2, c3 = cv2.split(converted_img)
    elif color_space == '-HSB':
        converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        c1, c2, c3 = cv2.split(converted_img)
    else:
        raise ValueError("Invalid color space option. Choose from -XYZ, -Lab, -YCrCb, -HSB.")
    
    return c1, c2, c3

def scale_to_grayscale(image_component):
    # Normalize the image component to fit in the range 0-255 for grayscale display
    return cv2.normalize(image_component, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

def convert_to_three_channel(image):
    # Convert a single-channel grayscale image to a three-channel grayscale image
    return cv2.merge([image, image, image])

def combine_images(images):
    # Combine four images into a 2x2 grid as per the required layout
    top_row = np.hstack((images[0], images[1]))
    bottom_row = np.hstack((images[2], images[3]))
    combined_image = np.vstack((top_row, bottom_row))
    return combined_image

def enforce_width_range(image, min_width=1280, max_width=1680):
    image_width = image.shape[1]
    if image_width > max_width:
        image = resize_image(image, max_width)
    elif image_width < min_width:
        image = resize_image(image, min_width)
    return image

# Analyzing and visualizing images in different color spaces
def visualize_image_color(color_space, image_file):
    image = cv2.imread(image_file)
    if image is None:
        print("Error: Unable to load image.")
        sys.exit(1)

    min_width = 1280
    max_width = 1680

    image = resize_image(image, max_width // 2)  # Resize to fit within half the max width for two images side by side

    c1, c2, c3 = convert_and_split_image(image, color_space)

    c1_gray = scale_to_grayscale(c1)
    c2_gray = scale_to_grayscale(c2)
    c3_gray = scale_to_grayscale(c3)

    combined_image = combine_images([image, convert_to_three_channel(c1_gray), convert_to_three_channel(c2_gray), convert_to_three_channel(c3_gray)])

    combined_image = enforce_width_range(combined_image, min_width, max_width)

    cv2.imshow("ChromaKey Result", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Using chroma keying to extract objects and composite them with backgrounds.
def apply_chromakey(scenic_image_file, green_screen_image_file):
    scenic_image = cv2.imread(scenic_image_file)
    green_screen_image = cv2.imread(green_screen_image_file)
    
    if scenic_image is None or green_screen_image is None:
        print("Error: Unable to load one or both images.")
        sys.exit(1)
    
    # Resize green screen image to fit within the width of the scenic image
    green_screen_image = resize_image(green_screen_image, scenic_image.shape[1])

    # Convert green screen image to HSB color space
    hsb_image = cv2.cvtColor(green_screen_image, cv2.COLOR_BGR2HSV)

    # Create mask to extract the person (green area filtering)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsb_image, lower_green, upper_green)

    # Invert the mask to get the person's silhouette
    person_mask = cv2.bitwise_not(mask)

    # Extract the person from the green screen image
    person = cv2.bitwise_and(green_screen_image, green_screen_image, mask=person_mask)

    # Create a white background
    white_background = np.ones_like(green_screen_image) * 255

    # Combine the person with the white background using the inverted mask
    person_with_white_bg = white_background.copy()
    person_with_white_bg[person_mask > 0] = person[person_mask > 0]

    # Prepare the final combined image
    height_diff = scenic_image.shape[0] - person.shape[0]
    translation_matrix = np.float32([[1, 0, (scenic_image.shape[1] - person.shape[1]) // 2], [0, 1, height_diff]])
    translated_person = cv2.warpAffine(person, translation_matrix, (scenic_image.shape[1], scenic_image.shape[0]))

    combined_scenic_person = scenic_image.copy()
    combined_scenic_person[translated_person > 0] = translated_person[translated_person > 0]

    # Combine all images for display
    combined_image = combine_images([green_screen_image, person_with_white_bg, scenic_image, combined_scenic_person])
    combined_image = enforce_width_range(combined_image)

    cv2.imshow("ChromaKey Result", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main function to handle both tasks
def main():
    if len(sys.argv) == 3 and sys.argv[1].startswith('-'):
        #Color space conversion
        color_space = sys.argv[1]
        image_file = sys.argv[2]
        visualize_image_color(color_space, image_file)
    elif len(sys.argv) == 3:
        #Green screen compositing
        scenic_image_file = sys.argv[1]
        green_screen_image_file = sys.argv[2]
        apply_chromakey(scenic_image_file, green_screen_image_file)
    else:
        print("Usage for Color space conversion: python ChromaKey.py -XYZ|-Lab|-YCrCb|-HSB imagefile")
        print("Usage for Green screen compositing: python ChromaKey.py scenicImageFile greenScreenImagefile")
        sys.exit(1)

if __name__ == "__main__":
    main()
