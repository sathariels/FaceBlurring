from operator import truediv

import cv2
import mediapipe as mp
import argparse
import os


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

args = argparse.ArgumentParser()

args.add_argument("--mode", default="webcam",)
args.add_argument("--filePath", default=None)
args = args.parse_args()



outputDir = './output'
if not os.path.exists(outputDir):
    os.makedirs(outputDir)


mpFaceDetection = mp.solutions.face_detection



with mpFaceDetection.FaceDetection(min_detection_confidence=.5, model_selection=0,) as faceDetection:

    if args.mode in ["image"]:
        img = cv2.imread(args.filePath)




        img = processImage(img, faceDetection)



        cv2.imwrite(os.path.join(outputDir, "output.png"), img)
    elif args.mode == "video":
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        outputVideo = cv2.VideoWriter(os.path.join(outputDir, 'output.mp4'),
                                      cv2.VideoWriter(*"MP4V"),
                                      25,
                                      (frame.shape[1], frame.shape[0]))
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()
        while ret:
            img = processImage(frame, faceDetection)

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
            cv2.waitKey(0)

            ret, frame = cap.read()


        cap.release()


