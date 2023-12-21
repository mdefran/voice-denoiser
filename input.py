import numpy as np
from tensorflow import keras
from scipy.io import wavfile
import librosa

sample_length = 512  # Update this based on your model's input size
sample_height = 1040  # Update this based on your model's input size

# Load the trained model
model = keras.models.load_model("denoiser.keras")

# Function to read a .wav file into a spectrogram
def wav_to_spectrogram(file_path, n_fft=1024, hop_length=512, save_phase=False):
    audio, sr = librosa.load(file_path, sr=None)
    
    # Compute the complex spectrogram with magnitude and phase information
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    # Do not return phase during training
    if save_phase:
        return spectrogram, sr, phase
    else:
        return spectrogram, sr

def preprocess_audio(file_path, sample_length, sample_height):
    # Convert the audio file to a spectrogram and save the phase
    spectrogram, sr, phase = wav_to_spectrogram(file_path, n_fft=2048, hop_length=512, save_phase=True)

    # Crop or pad the spectrogram and phase to the sample length and height
    cropped_spectrogram = np.pad(spectrogram[:, :sample_length], ((0, sample_height - spectrogram.shape[0]), (0, 0)), 'constant')
    cropped_phase = np.pad(phase[:, :sample_length], ((0, sample_height - phase.shape[0]), (0, 0)), 'constant')

    # Normalize the cropped spectrogram
    min_val = np.min(cropped_spectrogram)
    max_val = np.max(cropped_spectrogram)
    normalized_spectrogram = (cropped_spectrogram - min_val) / (max_val - min_val)

    return normalized_spectrogram, sr, cropped_phase, min_val, max_val

def reconstruct_audio(predicted_mask, original_spectrogram, phase, sr, min_val, max_val, hop_length=512):
    # Denormalized mask
    # predicted_mask = predicted_mask * (max_val - min_val) + min_val

    print(predicted_mask)

    # Apply the mask to the original spectrogram
    denoised_spectrogram = original_spectrogram * predicted_mask

    # Convert magnitude and phase to the complex spectrogram
    complex_spectrogram = denoised_spectrogram * np.exp(1j * phase)

    # Perform the inverse STFT
    reconstructed_audio = librosa.istft(complex_spectrogram, hop_length=hop_length)

    return reconstructed_audio

def save_wav_file(audio, sr, file_name):
    # Normalize the audio to the -1.0 to 1.0 range
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # Scale to 16-bit integer range and convert to integers
    scaled_audio = np.int16(audio * 32767)

    # Save the audio file
    wavfile.write(file_name, sr, scaled_audio)

new_file_path = "noisy_speech/noisy22_SNRdb_0.0_clnsp22.wav"

# Predicting on new data
normalized_spectrogram, sr, phase, min_val, max_val = preprocess_audio(new_file_path, sample_length, sample_height)

# Reshape the spectrogram for prediction
normalized_spectrogram_reshaped = normalized_spectrogram[np.newaxis, ..., np.newaxis]

# Predict the mask
predicted_mask = model.predict(normalized_spectrogram_reshaped)

# Post-process the prediction
predicted_mask = predicted_mask.squeeze()

# Reconstruct the audio using the original spectrogram and the predicted mask
reconstructed_audio = reconstruct_audio(predicted_mask, normalized_spectrogram, phase, min_val, max_val, sr)

# Save the reconstructed audio
save_wav_file(reconstructed_audio, sr, "output.wav")
