# Import the necessary libraries
import cv2  # OpenCV for image processing
import os   # For operating system operations like reading file names

# Define a class to handle bounding box annotations
class BBoxAnnotator:
    # Constructor to initialize various attributes
    def __init__(self, image_dir):
        self.image_dir = image_dir  # Directory where images are stored
        # List of all image files in the provided directory
        self.images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.current_image = None  # Image currently being processed
        self.bboxes = []  # List to store bounding box coordinates for the current image
        self.drawing = False  # Flag to indicate if we are currently drawing a bounding box
        self.current_bbox = []  # Temporary storage for the bounding box being drawn

    # Reset the drawing flag and current bounding box
    def reset(self):
        self.current_bbox = []
        self.drawing = False

    # Event handler for mouse events during annotation
    def draw_rectangle(self, event, x, y, flags, param):
        # On left button down, start drawing and store the starting coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_bbox = [(x, y)]
        # On left button up, finish drawing and store the ending coordinates
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_bbox.append((x, y))
            # Append the completed bounding box to the list
            self.bboxes.append(tuple(self.current_bbox))
            # Draw the rectangle on the image
            cv2.rectangle(self.current_image, self.current_bbox[0], self.current_bbox[1], (0, 255, 0), 2)
            # Display the updated image
            cv2.imshow('Image Annotation', self.current_image)
            # Reset for the next bounding box
            self.reset()

    # Main function to loop through each image and allow annotation
    def annotate(self):
        for image_path in self.images:
            # Load the current image
            self.current_image = cv2.imread(image_path)
            # Create a named window and set its mouse callback to our function
            cv2.namedWindow('Image Annotation')
            cv2.setMouseCallback('Image Annotation', self.draw_rectangle)
            # Display the current image and wait for user input (bounding boxes or next image)
            cv2.imshow('Image Annotation', self.current_image)
            cv2.waitKey(0)
            # Close the window
            cv2.destroyAllWindows()
            # Print the annotations for the current image
            print(f"Annotations for {image_path}: {self.bboxes}")
            # Reset the bounding boxes list for the next image
            self.bboxes = []

# If this script is executed directly, run the annotation process
if __name__ == '__main__':
    # Create an instance of the annotator
    annotator = BBoxAnnotator('/path/to/images/directory')
    # Start the annotation process
    annotator.annotate()
