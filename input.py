import numpy as np
from tensorflow import keras
from data import wav_to_spectrogram, save_spectrogram
from scipy.io import wavfile
import librosa

sample_length = 256

# Load the trained model
model = keras.models.load_model("denoiser.keras")

def spectrogram_to_wav(spectrogram, min_val, max_val, initial_phase, sr, n_fft=2048, hop_length=512, n_iter=100):
    # Denormalize the spectrogram
    spectrogram = (spectrogram * (max_val - min_val)) + min_val

    # Convert from mel back to original power
    spectrogram = librosa.db_to_power(spectrogram)

    # Get the approximate magnitude from the inverse
    magnitude = librosa.feature.inverse.mel_to_stft(spectrogram, sr=sr, n_fft=n_fft)

    # Combine magnitude and phase to form the complex spectrogram
    complex_spectrogram = magnitude * np.exp(1j * phase)

    # Perform the inverse STFT
    reconstructed_audio = librosa.istft(complex_spectrogram, hop_length=hop_length)

    return reconstructed_audio

def preprocess_audio(file_path):
    # Convert the audio file to a spectrogram and save the phase
    spectrogram, sr, magnitude, phase = wav_to_spectrogram(file_path, n_fft=2048, hop_length=512, save_phase=True)

    # Crop the spectrogram
    cropped_spectrogram = spectrogram[:, :sample_length]
    cropped_phase = phase[:, :sample_length]

    # Normalize the cropped spectrogram
    min_val = np.min(cropped_spectrogram)
    max_val = np.max(cropped_spectrogram)
    normalized_spectrogram = (cropped_spectrogram - min_val) / (max_val - min_val)

    return normalized_spectrogram, sr, cropped_phase, min_val, max_val

# Function to apply the mask and convert back to audio
def apply_mask_and_reconstruct(noisy_spectrogram, predicted_mask, sr, phase):
    # Apply the mask
    denoised_spectrogram = noisy_spectrogram * predicted_mask
    print(predicted_mask)

    save_spectrogram(denoised_spectrogram, sr, 512, "prediction.png")

    # Convert back to audio
    audio = spectrogram_to_wav(denoised_spectrogram, min_val, max_val, phase, sr)

    return audio

def save_wav_file(audio, sr, file_name):
    # Normalize the audio to the -1.0 to 1.0 range
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # Scale to 16-bit integer range and convert to integers
    scaled_audio = np.int16(audio * 32767)

    # Save the audio file
    wavfile.write(file_name, sr, scaled_audio)

# Predicting on new data
new_file_path = "noisy_speech/noisy38_SNRdb_0.0_clnsp38.wav"
new_spectrogram, sr, phase, min_val, max_val = preprocess_audio(new_file_path)

# Reshape the spectrogram for prediction (add batch and channel dimensions)
new_spectrogram = new_spectrogram[np.newaxis, ..., np.newaxis]

# Predict the mask
predicted_mask = model.predict(new_spectrogram)

# Post-process the prediction (remove batch dimension)
predicted_mask = predicted_mask.squeeze()
new_spectrogram = new_spectrogram.squeeze()

save_spectrogram(new_spectrogram, sr, 512, "input.png")

# Apply the mask and reconstruct the audio
denoised_audio = apply_mask_and_reconstruct(new_spectrogram, predicted_mask, sr, phase)

save_wav_file(denoised_audio, sr, "output.wav")