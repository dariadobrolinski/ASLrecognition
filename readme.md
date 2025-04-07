# ASL Recognition with Text-to-Speech

This project is an implementation of an American Sign Language (ASL) recognition system using computer vision and machine learning. The system recognizes ASL letters from a live webcam feed and speaks the recognized letters out loud using text-to-speech functionality.

This project is based on a [YouTube tutorial by Computer Vision Eng](https://youtu.be/MJCSjXepaAM?si=RB0kkmrQHyi-BM6M), but I added my own twist by integrating text-to-speech functionality to make the system more interactive and user-friendly.

---

## Features
- **Real-Time ASL Recognition**: Recognizes ASL letters from a live webcam feed.
- **Text-to-Speech Integration**: Speaks the recognized letters out loud using the `pyttsx3` library.
- **Customizable Dataset**: Allows users to collect their own ASL data for training.
- **Machine Learning Model**: Uses a Random Forest Classifier for letter recognition.
- **Hand Landmark Detection**: Utilizes MediaPipe Hands for detecting hand landmarks.

---

## How It Works
1. **Data Collection**:
   - Use `collect_images.py` to collect ASL data for each letter.
   - The script captures sequences of frames for each letter and saves them as `.npy` files.

2. **Dataset Creation**:
   - Use `create_dataset.py` to process the collected data.
   - Extracts hand landmarks from the frames and saves the processed data in a `data.pickle` file.

3. **Model Training**:
   - Use `train_classifier.py` to train a Random Forest Classifier on the processed dataset.
   - The trained model is saved as `model.p`.

4. **Real-Time Inference**:
   - Use `inference_classifier.py` to recognize ASL letters in real-time from a webcam feed.
   - The recognized letters are spoken out loud using the `pyttsx3` text-to-speech library.
