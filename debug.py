import cv2

# org = cv2.imread("samples/s0.png", -1)
# 
# ours = cv2.imread("samples/tiktok_mask.png", cv2.IMREAD_GRAYSCALE)
# 
# 
# print("a")

def generate_hole_mask(input_path, output_path, dilation_size=12):
    # Step 1: Load the image
    img = cv2.imread(input_path)

    # Step 2: Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Threshold the grayscale image
    _, binary = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY_INV)

    # Step 4: Perform dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_size, dilation_size))
    dilated = cv2.dilate(binary, kernel)

    # Save the image
    cv2.imwrite(output_path, dilated * 255)


if __name__ == "__main__":
    generate_hole_mask("dCOCO/im1_raw/95022_00.jpg", "image_mask.png")