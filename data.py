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

clean_speech_folder = "clean_speech"
noisy_speech_folder = "noisy_speech"
snr_intervals = 5

# Function to read a .wav file into a spectrogram
def wav_to_spectrogram(file_path, n_fft=2048, hop_length=512, save_phase=False):
    audio, sr = librosa.load(file_path, sr=None)
    
    # Compute the complex spectrogram with magnitude and phase information
    stft_matrix = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    # Compute the mel spectrogram
    spectrogram = librosa.feature.melspectrogram(S=magnitude**2, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Only return spectrogram during training
    if save_phase:
        return spectrogram, sr, magnitude, phase
    else:
        return spectrogram, sr

# Function to convert a spectrogram back into a .wav file using saved phase information
def spectrogram_to_wav(spectrogram, magnitude, phase, sr, hop_length=512):
    # Reconstruct the complex spectrogram using saved magnitude and phase
    reconstructed_complex_spectrogram = magnitude * np.exp(1j * phase)

    # Perform the iSTFT to obtain the reconstructed audio
    reconstructed_audio = librosa.istft(reconstructed_complex_spectrogram, hop_length=hop_length)

    return reconstructed_audio

def save_spectrogram(spectrogram, sr, hop_length, save_path):
    plt.figure(figsize=(10, 4))

    # Reshape the spectrogram if it is 1D
    if len(spectrogram.shape) == 1:
        spectrogram = np.expand_dims(spectrogram, axis=0)

    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(save_path)
    plt.close()

def save_mask(mask, file_name):
    plt.imsave(file_name, mask, cmap="gray")

def load_data(clean_folder, noisy_folder):
    clean_spectrograms = []
    noisy_spectrograms = []

    # Load the clean speech files
    for clean_file in os.listdir(clean_folder):
        if clean_file.endswith(".wav"):
            # Load the clean speech file
            clean_path = os.path.join(clean_folder, clean_file)
            clean_spectrogram, _ = wav_to_spectrogram(clean_path)

            # Add the clean file multiple times for all corresponding noisy SNR intervals
            for i in range(snr_intervals):
                clean_spectrograms.append(clean_spectrogram)

    # Load the noisy speech files
    for noisy_file in os.listdir(noisy_folder):
        if noisy_file.endswith(".wav"):
            # Load the noisy speech file
            noisy_path = os.path.join(noisy_folder, noisy_file)
            noisy_spectrogram, _ = wav_to_spectrogram(noisy_path)
            noisy_spectrograms.append(noisy_spectrogram)

    # Find the largest valid spectrogram length for data and model dimensionality requirements
    min_length = min([s.shape[1] for s in (clean_spectrograms + noisy_spectrograms)])
    sample_length = 1 << (min_length.bit_length() - 1) if min_length > 1 else 0

    # Crop spectrograms to have consistent length
    for i in range(len(clean_spectrograms)):
        clean_spectrograms[i] = clean_spectrograms[i][:, :sample_length]
        noisy_spectrograms[i] = noisy_spectrograms[i][:, :sample_length]

    # Convert the lists to numpy arrays
    clean_spectrograms = np.array(clean_spectrograms)
    noisy_spectrograms = np.array(noisy_spectrograms)

    # Create binary masks out of the clean spectrograms
    clean_spectrograms = np.where(clean_spectrograms >= -60, 1, 0)

    # Save images for debugging
    for i in range(len(clean_spectrograms)):
        save_spectrogram(noisy_spectrograms[i], 16000, 512, f"saved_spects/{i}")
        save_mask(clean_spectrograms[i], f"saved_masks/{i}.png")

    # Apply normalization to the noisy spectrograms
    for i in range(len(noisy_spectrograms)):
        min_val = np.min(noisy_spectrograms[i])
        max_val = np.max(noisy_spectrograms[i])
        noisy_spectrograms[i] = (noisy_spectrograms[i] - min_val) / (max_val - min_val)

    return clean_spectrograms, noisy_spectrograms

def unet_model(input_shape):
    inputs = keras.Input(input_shape)

    # Downsample
    x = layers.Conv2D(64, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical: downsampling
    for filters in [128, 256, 512]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Upsample
    for filters in [512, 256, 128, 64]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)

        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

# Load the data
clean_spectrograms, noisy_spectrograms = load_data("clean_speech", "noisy_speech")
width, height = noisy_spectrograms.shape[1], noisy_spectrograms.shape[2]

# Split the data into training and validation sets
noisy_train, noisy_val, clean_train, clean_val = train_test_split(
    noisy_spectrograms, clean_spectrograms, test_size = 0.2, random_state = 42
)
print(clean_spectrograms[1])

# Create and compile the UNet model
model = unet_model((width, height, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model to the data
model.fit(
    noisy_train,
    clean_train,
    epochs = 10,
    batch_size = 16,
    validation_data = (noisy_val, clean_val)
)

model.save("denoiser.keras")