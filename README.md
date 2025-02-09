# Speech Emotion Recognition

## Overview
This project is a real-time speech emotion recognition system built using deep learning. It uses LSTM neural networks to classify emotions from speech audio, featuring both real-time recording and file upload capabilities.

## Features
- 🎤 Real-time emotion recognition from speech
- 📁 Support for WAV file uploads
- 📊 Visual display of model metrics
- 📈 Training history visualization
- 🎯 Confusion matrix analysis
- 🖥️ User-friendly GUI interface

## Dataset
This project uses the Toronto Emotional Speech Set (TESS):
- **Source**: [TESS Dataset on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- **Emotions**: Angry, Disgust, Fear, Happy, Neutral, Pleasant Surprise, Sad
- **Format**: WAV audio files
- **Sample Rate**: 22050 Hz

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application
```bash
python src/emotion_recognition.py
```

### GUI Features
- **Record**: Click to start/stop recording
- **Load Audio**: Upload WAV files for analysis
- **Results**: View predicted emotions
- **Metrics**: Monitor model performance

## Technical Details

### Model Architecture
- Type: LSTM Neural Network
- Input: 40 MFCC features
- Layers:
  - LSTM (256 units)
  - LSTM (128 units)
  - LSTM (64 units)
  - Dense (32 units)
  - Output (7 emotions)

### Performance
- Training epochs: 50
- Batch size: 32
- Validation split: 20%
- Model metrics displayed in GUI

## Project Structure
```
emotion-recognition/
├── src/
│   └── emotion_recognition.py    # Main application
├── data/
│   └── DataSetSpeech/           # Dataset directory
├── models/
│   └── emotion_model11.h5       # Trained model
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## Dependencies
- numpy
- librosa
- soundfile
- scikit-learn
- tensorflow
- sounddevice
- matplotlib
- seaborn
- tkinter

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Toronto Emotional Speech Set (TESS) Dataset creators
- Libraries and tools used in this project
- Contributors and maintainers

## Contact
Your Name - [@rozhin77](https://github.com/rozhin77)
Project Link: [https://github.com/rozhin77/emotion-recognition](https://github.com/rozhin77/emotion-recognition)

---
⭐ Don't forget to star this repo if you found it helpful!