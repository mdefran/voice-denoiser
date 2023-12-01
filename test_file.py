import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# Function to read a .wav file into a spectrogram
def wav_to_spectrogram(file_path, n_fft=2048, hop_length=512, save_phase=False):
    audio, sr = librosa.load(file_path, sr=None)
    
    # Compute the complex spectrogram with magnitude and phase information
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    # Compute the mel spectrogram
    spectrogram = librosa.feature.melspectrogram(S=magnitude**2, sr=sr, n_fft=n_fft, hop_length=hop_length)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Only return spectrogram during training
    if save_phase:
        return log_spectrogram, sr, magnitude, phase
    else:
        return log_spectrogram, sr

# Function to convert a spectrogram back into a .wav file using saved phase information
def spectrogram_to_wav(spectrogram, magnitude, phase, sr, hop_length=512):
    # Reconstruct the complex spectrogram using saved magnitude and phase
    reconstructed_complex_spectrogram = magnitude * np.exp(1j * phase)

    # Perform the iSTFT to obtain the reconstructed audio
    reconstructed_audio = librosa.istft(reconstructed_complex_spectrogram, hop_length=hop_length)

    return reconstructed_audio

def display_spectrogram(spectrogram, sr, hop_length, save_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(save_path)

def load_data(clean_speech_folder, noisy_speech_folder, masks_folder, data_folder):
    db_interval = -60
    max_height = 0
    max_width = 0

    # Find the dimensions of the largest image for padding
    for clean_file_name in os.listdir(clean_speech_folder):
        if clean_file_name.endswith(".wav"):
            clean_speech_path = os.path.join(clean_speech_folder, clean_file_name)
            clean_speech_spectrogram, _ = wav_to_spectrogram(clean_speech_path)
            height, width = clean_speech_spectrogram.shape
            max_height = max(max_height, height)
            max_width = max(max_width, width)

    # Iterate through source files
    for clean_file_name in os.listdir(clean_speech_folder):
        if clean_file_name.endswith(".wav"):
            # Load the clean speech file
            clean_speech_path = os.path.join(clean_speech_folder, clean_file_name)
            clean_speech_spectrogram, _ = wav_to_spectrogram(clean_speech_path)

            # Calculate padding
            pad_height = max_height - clean_speech_spectrogram.shape[0]
            pad_width = max_width - clean_speech_spectrogram.shape[1]

            # Add padding to the clean speech spectrogram
            clean_speech_spectrogram = np.pad(clean_speech_spectrogram, ((0, pad_height), (0, pad_width)), mode='constant')

            # Create a binary mask from the clean speech file
            mask = (clean_speech_spectrogram > db_interval).astype(np.float32)

            # Iterate through files with corresponding clean speech file at all SNRs
            for noisy_file_name in os.listdir(noisy_speech_folder):
                if noisy_file_name.endswith(clean_file_name):
                    # Load the noisy speech file
                    noisy_speech_path = os.path.join(noisy_speech_folder, noisy_file_name)
                    noisy_speech_spectrogram, _ = wav_to_spectrogram(noisy_speech_path)

                    # Add padding to the noisy speech spectrogram
                    noisy_speech_spectrogram = np.pad(noisy_speech_spectrogram, ((0, pad_height), (0, pad_width)), mode='constant')

                    # Save the noisy spectrogram data
                    output_file_name = noisy_file_name.replace(".wav", "_spectrogram.png")
                    output_path = os.path.join(data_folder, output_file_name)
                    plt.imsave(output_path, noisy_speech_spectrogram, cmap='gray')

                    # Save the binary mask
                    output_file_name = noisy_file_name.replace(".wav", "_mask.png")
                    output_path = os.path.join(masks_folder, output_file_name)
                    plt.imsave(output_path, mask, cmap='gray')

    return max_width, max_height

def unet_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Define the encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Define the decoder
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    up1 = layers.UpSampling2D(size=(2, 2))(conv2)

    # Add more encoder-decoder blocks as needed

    outputs = layers.Conv2D(1, 1, activation='softmax')(up1)
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

width, height = load_data("clean_speech", "noisy_speech", "masks", "data")
data_train, data_val, mask_train, mask_val = train_test_split(
    data, masks test_size=0.2, random_state=42
)

model = unet_model((width, height, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit()