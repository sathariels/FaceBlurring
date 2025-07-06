from operator import truediv
import cv2
import mediapipe as mp
import argparse
import os


class FaceBlurProcessor:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """Initialize the face blur processor"""
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )
        self.outputDir = './output'
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)

    def processImage(self, img):
        """Process a single image and blur detected faces"""
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.faceDetection.process(imgRgb)
        H, W, _ = img.shape

        if faces.detections is not None:
            for detections in faces.detections:
                locationData = detections.location_data
                bBox = locationData.relative_bounding_box

                x1, y1, w, h = bBox.xmin, bBox.ymin, bBox.width, bBox.height

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                w = int(w * W)
                h = int(h * H)

                img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (60, 60))

        return img

    def process_image_file(self, file_path):
        """Process an image file and save the blurred version"""
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Could not read image: {file_path}")

        # Extract filename without extension and create output name
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_blurred{ext}"

        # Process the image
        processed_img = self.processImage(img)

        # Save the processed image
        output_path = os.path.join(self.outputDir, output_filename)
        cv2.imwrite(output_path, processed_img)

        return output_path

    def process_video_file(self, file_path):
        """Process a video file and save the blurred version"""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {file_path}")

        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read video: {file_path}")

        # Extract filename without extension and create output name
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_blurred.mp4"

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25  # Default FPS if detection fails

        # Setup output video
        output_path = os.path.join(self.outputDir, output_filename)
        outputVideo = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frame.shape[1], frame.shape[0])
        )

        # Reset video capture
        cap.release()
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()

        # Process all frames
        while ret:
            processed_frame = self.processImage(frame)
            outputVideo.write(processed_frame)
            ret, frame = cap.read()

        # Cleanup
        cap.release()
        outputVideo.release()

        return output_path

    def start_webcam(self):
        """Start webcam processing with face blur"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("Could not open webcam")

        ret, frame = cap.read()
        while ret:
            processed_frame = self.processImage(frame)

            cv2.imshow("Webcam - Press 'q' to quit", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()


# Original script functionality (for command line usage)
def processImage(img, faceDetection):
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = faceDetection.process(imgRgb)
    H, W, _ = img.shape
    if faces.detections is not None:
        for detections in faces.detections:
            locationData = detections.location_data
            bBox = locationData.relative_bounding_box

            x1, y1, w, h = bBox.xmin, bBox.ymin, bBox.width, bBox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (60, 60))
    return img


def main():
    """Main function for command line usage (your original script)"""
    args = argparse.ArgumentParser()
    args.add_argument("--mode", default="webcam")
    args.add_argument("--filePath", default=None)
    args = args.parse_args()

    outputDir = './output'
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    mpFaceDetection = mp.solutions.face_detection

    with mpFaceDetection.FaceDetection(min_detection_confidence=.5, model_selection=0) as faceDetection:
        if args.mode in ["image"]:
            img = cv2.imread(args.filePath)

            # Extract filename without extension and create output name
            filename = os.path.basename(args.filePath)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_blurred{ext}"

            img = processImage(img, faceDetection)

            cv2.imwrite(os.path.join(outputDir, output_filename), img)

        elif args.mode == "video":
            cap = cv2.VideoCapture(args.filePath)
            ret, frame = cap.read()

            # Extract filename without extension and create output name
            filename = os.path.basename(args.filePath)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_blurred.mp4"  # Keep as .mp4 for video

            outputVideo = cv2.VideoWriter(os.path.join(outputDir, output_filename),
                                          cv2.VideoWriter_fourcc(*"mp4v"),  # Fixed this line
                                          25,
                                          (frame.shape[1], frame.shape[0]))
            cap = cv2.VideoCapture(args.filePath)
            ret, frame = cap.read()
            while ret:
                frame = processImage(frame, faceDetection)  # Fixed: was using img instead of frame

                outputVideo.write(frame)

                ret, frame = cap.read()
            cap.release()
            outputVideo.release()

        elif args.mode in ["webcam"]:
            cap = cv2.VideoCapture(0)

            ret, frame = cap.read()
            while ret:
                frame = processImage(frame, faceDetection)

                cv2.imshow("Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Fixed: was waitKey(0) which pauses indefinitely
                    break

                ret, frame = cap.read()

            cap.release()
            cv2.destroyAllWindows()  # Added this to properly close windows


if __name__ == "__main__":
    main()