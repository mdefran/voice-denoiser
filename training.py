import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.io import wavfile

def get_spectrogram(file_path, n_fft=2048, hop_length=512):
    """
    Compute the mel spectrogram of an audio file.

    Parameters:
    - file_path (str): The path to the audio file.
    - n_fft (int, optional): Number of FFT points. Default is 2048.
    - hop_length (int, optional): Number of samples between successive frames. Default is 512.

    Returns:
    - log_spectrogram (numpy.ndarray): Log-scaled mel spectrogram of the audio.
    """
    audio, _ = librosa.load(file_path, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=_, n_fft=n_fft, hop_length=hop_length)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram

def get_label(filename):
    if "clnsp" in filename and "noisy" in filename:
        return "noisyspeech"
    elif "clnsp" in filename:
        return "clean"
    elif "noisy" in filename:
        return "noisy"
    else:
        return "unknown"

def load_data(folder_path):
    spectrograms = []
    labels = []
    min_length = float('inf')

    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            
            label = get_label(filename)
            spectrogram = get_spectrogram(file_path)

            min_length = min(min_length, spectrogram.shape[1])

            spectrograms.append(spectrogram)
            labels.append(label)

    cropped_spectrograms = [spec[:, :min_length] for spec in spectrograms]

    return np.array(cropped_spectrograms), np.array(labels)

def save_phase_information(file_path, output_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Compute the complex spectrogram (magnitude and phase)
    stft_matrix = librosa.stft(audio)
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    # Save the magnitude and phase information
    np.save(output_path + "_magnitude.npy", magnitude)
    np.save(output_path + "_phase.npy", phase)

def reconstruct_audio(magnitude_path, phase_path, output_path):
    # Load the magnitude and phase information
    magnitude = np.load(magnitude_path)
    phase = np.load(phase_path)

    # Combine magnitude and phase to obtain the complex spectrogram
    complex_spectrogram = magnitude * np.exp(1j * phase)

    # Perform the inverse Short-Time Fourier Transform (iSTFT) to obtain the reconstructed audio
    reconstructed_audio = librosa.istft(complex_spectrogram)

    # Save the reconstructed audio as a new .wav file
    wavfile.write(output_path, 44100, reconstructed_audio)

folder_path = "MS-SNSD/NoisySpeech_training"
spectrograms, labels = load_data(folder_path)
print(f"Loaded {len(spectrograms)} spectrograms from the folder.")

reconstructed_audio = librosa.istft(librosa.db_to_power(spectrograms[1]), hop_length=512)
output_file_path = "separated/re.wav"
write(output_file_path, 22050, reconstructed_audio)

save_phase_information("CleanSpeech_training/clnsp1.wav", "test.wav")

# Example usage to reconstruct audio using saved phase information
reconstruct_audio("output_path_magnitude.npy", "output_path_phase.npy", "output_path_reconstructed.wav")