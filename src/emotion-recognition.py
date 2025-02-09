#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import librosa
import soundfile
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import sounddevice as sd
import queue
import wave
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class EmotionRecognizer:
    def __init__(self):
        self.model = None
        self.recording = False
        self.audio_queue = queue.Queue()
        self.sample_rate = 22050
        self.setup_gui()
        self.load_or_train_model()

    def setup_gui(self):
        self.window = tk.Tk()
        self.window.title("Emotion Recognition")
        self.window.geometry("800x600")  

        # Main frame
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Control buttons frame
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        tk.Button(control_frame, text="Record", command=self.toggle_recording).pack(pady=10)
        tk.Button(control_frame, text="Load Audio File", command=self.load_audio_file).pack(pady=10)
        
        # Result label
        self.result_label = tk.Label(control_frame, text="Result: ")
        self.result_label.pack(pady=10)

        # Metrics frame
        metrics_frame = tk.Frame(main_frame)
        metrics_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Metrics display
        self.metrics_text = tk.Text(metrics_frame, height=20, width=50)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        # Add scrollbar
        scrollbar = tk.Scrollbar(metrics_frame, command=self.metrics_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.metrics_text.config(yscrollcommand=scrollbar.set)

        # Image display
        self.image_label = tk.Label(metrics_frame)
        self.image_label.pack(pady=10)

    def load_or_train_model(self):
        model_path = "emotion_model11.h5"
        if os.path.exists(model_path):
            print("Loading existing model...")
            self.model = load_model(model_path)
            # Load saved metrics if available
            if os.path.exists('model_metrics.npy'):
                self.metrics = np.load('model_metrics.npy', allow_pickle=True).item()
                self.update_metrics_display()
        else:
            print("Training new model...")
            self.train_model()
            self.update_metrics_display()

    def extract_features(self, audio_path):
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, duration=3, offset=0.5, sr=self.sample_rate)
            
            # Extract features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            
            return mfcc_scaled
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {str(e)}")
            return None

    def train_model(self):
        data_dir = r"C:\Users\hP\DataSetSpeech"  
        print(f"Looking for audio files in: {data_dir}")

        if not os.path.exists(data_dir):
            raise ValueError(f"Directory not found: {data_dir}")

        emotion_dict = {
            'OAF_angry': 0,
            'OAF_disgust': 1,
            'OAF_Fear': 2,
            'OAF_happy': 3,
            'OAF_neutral': 4,
            'OAF_Pleasant_surprise': 5,
            'OAF_Sad': 6,
            'YAF_angry': 0,
            'YAF_disgust': 1,
            'YAF_fear': 2,
            'YAF_happy': 3,
            'YAF_neutral': 4,
            'YAF_pleasant_surprised': 5,
            'YAF_sad': 6
        }

        X = []
        y = []

        print("\nProcessing directories...")
        for emotion_folder in emotion_dict.keys():
            folder_path = os.path.join(data_dir, emotion_folder)
            if os.path.exists(folder_path):
                print(f"\nProcessing {emotion_folder}")
                for filename in os.listdir(folder_path):
                    if filename.endswith('.wav'):
                        file_path = os.path.join(folder_path, filename)
                        features = self.extract_features(file_path)
                        if features is not None:
                            X.append(features)
                            emotion_label = emotion_dict[emotion_folder]
                            y.append(emotion_label)
                            print(f"Processed: {filename}")

        if len(X) == 0:
            raise ValueError("No audio files were loaded. Check your data directory.")

        X = np.array(X)
        y = np.array(y)
        y = to_categorical(y)

        print("\nDataset shape:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train model
        self.model = Sequential([
            LSTM(256, input_shape=(X_train.shape[1], 1), return_sequences=True),
            Dropout(0.2),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(y.shape[1], activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Reshape data for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=32
        )

        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # Calculate confusion matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        # Create classification report
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Pleasant Surprise', 'Sad']
        report = classification_report(y_test_classes, y_pred_classes, target_names=emotion_labels)
        
        # Plot training history
        self.plot_training_history(history)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, emotion_labels)
        
        # Save model
        self.model.save("emotion_model11.h5")
        
        # Store metrics for GUI display
        self.metrics = {
            'accuracy': test_accuracy,
            'history': history.history,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        # Save metrics
        np.save('model_metrics.npy', self.metrics)

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    def plot_confusion_matrix(self, cm, labels):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

    def update_metrics_display(self):
        if hasattr(self, 'metrics'):
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, f"Model Accuracy: {self.metrics['accuracy']*100:.2f}%\n\n")
            self.metrics_text.insert(tk.END, "Classification Report:\n")
            self.metrics_text.insert(tk.END, self.metrics['classification_report'])

            # Load and display training history plot
            if os.path.exists('training_history.png'):
                img = tk.PhotoImage(file='training_history.png')
                self.image_label.config(image=img)
                self.image_label.image = img  # Keep a reference!

    def predict_emotion(self, audio_path):
        features = self.extract_features(audio_path)
        if features is not None:
            features = features.reshape(1, features.shape[0], 1)
            prediction = self.model.predict(features)
            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Pleasant Surprise', 'Sad']
            predicted_emotion = emotion_labels[np.argmax(prediction)]
            return predicted_emotion
        return "Error processing audio"

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.audio_data = []
            threading.Thread(target=self.record_audio).start()
        else:
            self.recording = False
            self.save_and_process_recording()

    def record_audio(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_queue.put(indata.copy())

        with sd.InputStream(callback=callback, channels=1, samplerate=self.sample_rate):
            while self.recording:
                self.audio_data.extend(self.audio_queue.get().flatten())

    def save_and_process_recording(self):
        if not self.audio_data:
            return

        temp_file = "temp_recording.wav"
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(np.array(self.audio_data).tobytes())

        emotion = self.predict_emotion(temp_file)
        self.result_label.config(text=f"Predicted Emotion: {emotion}")
        os.remove(temp_file)

    def load_audio_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            emotion = self.predict_emotion(file_path)
            self.result_label.config(text=f"Predicted Emotion: {emotion}")

    def run(self):
        self.window.mainloop()

def main():
    app = EmotionRecognizer()
    app.run()

if __name__ == "__main__":
    main()


# In[ ]:




