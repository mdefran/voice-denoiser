import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras import layers, models

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

    # Only return spectrogram for training
    if save_phase:
        return log_spectrogram, sr, magnitude, phase
    else:
        return log_spectrogram

# Function to convert a spectrogram back into a .wav file using saved phase information
def spectrogram_to_wav(spectrogram, magnitude, phase, sr, hop_length=512):
    # Reconstruct the complex spectrogram using saved magnitude and phase
    reconstructed_complex_spectrogram = magnitude * np.exp(1j * phase)

    # Perform the iSTFT to obtain the reconstructed audio
    reconstructed_audio = librosa.istft(reconstructed_complex_spectrogram, hop_length=hop_length)

    return reconstructed_audio


# Function to fetch data and prepare it for training
def load_and_preprocess_data(clean_folder, noise_folder, noisy_folder):
    source_data = []
    mixed_data = []
    labels = []
    max_frames = 0

    # Load clean speech data
    for filename in os.listdir(clean_folder):
        if filename.endswith(".wav"):
            clean_path = os.path.join(clean_folder, filename)
            file_number = filename[5:-4]
            noisy_filenames = [f"noisy{file_number}_SNRdb_{snr}_clnsp{file_number}.wav" for snr in [0.0, 10.0, 20.0, 30.0, 40.0]]

            for noisy_filename in noisy_filenames:
                noisy_path = os.path.join(noisy_folder, noisy_filename)

                clean_spec = wav_to_spectrogram(clean_path)
                noisy_spec = wav_to_spectrogram(noisy_path)

                source_data.append(clean_spec)
                mixed_data.append(noisy_spec)
                labels.append(1)  # Label 1 for clean speech

                # Update max_frames if necessary
                max_frames = max(max_frames, clean_spec.shape[1], noisy_spec.shape[1])

    # Load background noise data
    for filename in os.listdir(noise_folder):
        if filename.endswith(".wav"):
            noise_path = os.path.join(noise_folder, filename)

            noise_spec = wav_to_spectrogram(noise_path)

            source_data.append(noise_spec)
            mixed_data.append(noise_spec)
            labels.append(0)  # Label 0 for background noise

            # Update max_frames if necessary
            max_frames = max(max_frames, noise_spec.shape[1])

    # Perform dynamic padding
    source_data = [np.pad(spec, ((0, 0), (0, max_frames - spec.shape[1]))) for spec in source_data]
    mixed_data = [np.pad(spec, ((0, 0), (0, max_frames - spec.shape[1]))) for spec in mixed_data]

    source_data = np.array(source_data)
    mixed_data = np.array(mixed_data)
    labels = np.array(labels)

    return source_data, mixed_data, labels

# Read files into lists
clean_folder = "MS-SNSD/CleanSpeech_training"
noise_folder = "MS-SNSD/Noise_training"
noisy_folder = "MS-SNSD/NoisySpeech_training"
source_data, mixed_data, labels = load_and_preprocess_data(clean_folder, noise_folder, noisy_folder)

index_to_visualize = 0

# Plot the spectrogram
plt.figure(figsize=(10, 4))
plt.imshow(source_data[index_to_visualize], cmap='viridis', aspect='auto', origin='lower')
plt.title(f"Spectrogram for Index {index_to_visualize}")
plt.xlabel("Time Frames")
plt.ylabel("Frequency Bins")
plt.colorbar(format="%+2.0f dB")
plt.savefig('spectrogram_plot.png')  # Save the plot to a file