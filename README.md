# Real-Time Emotion Detection

Detects faces from a webcam feed with [MediaPipe](https://developers.google.com/mediapipe) and classifies each face's emotion with a fine-tuned ResNet-50 (PyTorch).

Recognized emotions: **Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral**.

## How it works

1. **Training** – An ImageNet-pretrained ResNet-50 is fine-tuned on a 7-class facial-expression dataset (new classification head with dropout, Adam, `ReduceLROnPlateau`, mixed precision, early stopping). The best checkpoint by validation loss is saved to `best_model.pt`.
2. **Inference** – MediaPipe finds the face in each webcam frame, the face is cropped and passed to the model every 5th frame, and the prediction is drawn over the video. Predictions below 75% confidence are shown as "Uncertain".

All code lives in [`FaceDetection.ipynb`](FaceDetection.ipynb).

## Setup

Python 3.10+ is recommended. A CUDA GPU is needed for training as written (it uses `torch.amp` with CUDA); webcam inference runs on CPU or GPU.

```bash
git clone <your-repo-url>
cd emotion-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook FaceDetection.ipynb
```

## Dataset

Training data is not included. Place it in this layout (`ImageFolder` format):

```
DATASET/
├── train/
│   ├── <class folders>/
│   └── ...
└── test/
    ├── <class folders>/
    └── ...
```

`ImageFolder` assigns class indices by sorting folder names alphabetically, so the folders must sort into the same order as the label map used in the webcam cell:

| Index | Emotion   |
|-------|-----------|
| 0     | Surprise  |
| 1     | Fear      |
| 2     | Disgust   |
| 3     | Happiness |
| 4     | Sadness   |
| 5     | Anger     |
| 6     | Neutral   |

## Usage

**Train**: the training and evaluation cells in the notebook are wrapped in `'''` triple quotes. Remove the quotes to run them. Training stops early based on validation loss and writes `best_model.pt`.

**Run on webcam**: make sure `best_model.pt` is in the project folder, then run the cells under *Web Cam*. Press `q` in the video window to quit.

## Notes

- Emotion classifiers are sensitive to lighting, camera angle, and demographics of the training set. Treat predictions as a demo, not ground truth.
- Only the first detected face is classified, although the face count shows all detected faces.

## License

[MIT](LICENSE)
