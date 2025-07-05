# Face Blur Detection

A Python application that automatically detects and blurs faces in images, videos, and live webcam feeds using OpenCV and MediaPipe.

## Features

- **Multiple Input Modes**: Process static images, video files, or live webcam streams
- **Automatic Face Detection**: Powered by MediaPipe's robust face detection model
- **Real-time Processing**: Live webcam face blurring with instant preview
- **Batch Video Processing**: Automatically process entire video files
- **Privacy Protection**: Applies strong Gaussian blur to detected faces
- **Auto Output Management**: Creates output directory and saves results automatically

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd face-blur-detection
```

2. Install required dependencies:
```bash
pip install opencv-python mediapipe
```

## Usage

### Image Processing
Process a single image file:
```bash
python face_blur.py --mode image --filePath path/to/your/image.jpg
```

### Video Processing
Process a video file:
```bash
python face_blur.py --mode video --filePath path/to/your/video.mp4
```

### Live Webcam (Default)
Start live webcam processing:
```bash
python face_blur.py --mode webcam
```
or simply:
```bash
python face_blur.py
```

## Command Line Arguments

- `--mode`: Processing mode (`image`, `video`, or `webcam`)
- `--filePath`: Path to input file (required for image and video modes)

## Output

- **Images**: Saved as `output.png` in the `./output` directory
- **Videos**: Saved as `output.mp4` in the `./output` directory
- **Webcam**: Live preview window (press any key to advance frames)

The output directory is automatically created if it doesn't exist.

## How It Works

1. **Face Detection**: Uses MediaPipe's face detection solution to identify faces
2. **Coordinate Conversion**: Converts relative bounding box coordinates to absolute pixels
3. **Blur Application**: Applies a 60x60 Gaussian blur kernel to detected face regions
4. **Output Generation**: Saves processed results or displays them in real-time

## Configuration

You can modify these parameters in the code:

- **Detection Confidence**: Currently set to 0.5 (50% confidence threshold)
- **Blur Intensity**: 60x60 Gaussian blur kernel size
- **Model Selection**: Uses MediaPipe's standard face detection model (model_selection=0)

## Use Cases

- Privacy protection for social media content
- Content moderation for user-generated media
- Automatic anonymization of surveillance footage
- Batch processing of photo/video collections
- Real-time privacy filtering for video calls

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
