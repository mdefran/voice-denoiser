import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras import layers, models

clean_speech_folder = "clean_speech"
noisy_speech_folder = "noisy_speech"
snr_intervals = 5

def wav_to_spectrogram(file_path, n_fft=1024, hop_length=512, save_phase=False):
    """
    Convert a .wav file to a spectrogram.

    Parameters:
    - file_path (str): Path to the .wav file.
    - n_fft (int): Number of FFT components.
    - hop_length (int): Number of audio samples between adjacent STFT columns.
    - save_phase (bool): Whether to return the phase information.

    Returns:
    - tuple: A tuple containing the spectrogram and sampling rate. If save_phase is True, also includes the phase.
    """
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

def save_spectrogram(spectrogram, sr, n_fft, hop_length, save_path):
    """
    Save a cropped spectrogram as an image file.

    Parameters:
    - spectrogram (ndarray): Cropped spectrogram data to be saved.
    - sr (int): Sampling rate of the audio used to create the original spectrogram.
    - n_fft (int): Number of FFT components used to create the original spectrogram.
    - hop_length (int): Hop length used in the STFT.
    - save_path (str): Path where the image will be saved.
    """
    plt.figure(figsize=(10, 4))

    # Reshape the spectrogram if it is 1D
    if len(spectrogram.shape) == 1:
        spectrogram = np.expand_dims(spectrogram, axis=0)

    # Calculate the frequency scale for the cropped spectrogram
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    cropped_freqs = freqs[:spectrogram.shape[0]]

    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', fmin=cropped_freqs[0], fmax=cropped_freqs[-1])
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(save_path)
    plt.close()


def save_mask(mask, file_name):
    """
    Save a binary mask image.

    Parameters:
    - mask (ndarray): Binary mask data.
    - file_name (str): Name or path of the file where the mask will be saved.
    """
    plt.imsave(file_name, mask, cmap="gray")

def load_data(clean_folder, noisy_folder):
    """
    Load and preprocess spectrogram data from clean and noisy speech files.

    Parameters:
    - clean_folder (str): Directory containing clean speech .wav files.
    - noisy_folder (str): Directory containing noisy speech .wav files.

    Returns:
    - tuple: A tuple containing arrays of noisy spectrograms and corresponding binary masks.
    """
    clean_spectrograms = []
    noisy_spectrograms = []

    # Load the clean speech files
    for clean_file in os.listdir(clean_folder):
        if clean_file.endswith(".wav"):
            clean_path = os.path.join(clean_folder, clean_file)
            clean_spectrogram, _ = wav_to_spectrogram(clean_path)

            # Add the clean file multiple times for all corresponding noisy SNR intervals
            for i in range(snr_intervals):
                clean_spectrograms.append(clean_spectrogram)

    # Load the noisy speech files
    for noisy_file in os.listdir(noisy_folder):
        if noisy_file.endswith(".wav"):
            noisy_path = os.path.join(noisy_folder, noisy_file)
            noisy_spectrogram, _ = wav_to_spectrogram(noisy_path)
            noisy_spectrograms.append(noisy_spectrogram)

    # Find the largest valid spectrogram time length for model dimensionality requirements
    min_length = min([s.shape[1] for s in (clean_spectrograms + noisy_spectrograms)])
    sample_length = 1 << (min_length.bit_length() - 1) if min_length > 1 else 0

    # Crop spectrogram times to have consistent length
    for i in range(len(clean_spectrograms)):
        clean_spectrograms[i] = clean_spectrograms[i][:, :sample_length]
        noisy_spectrograms[i] = noisy_spectrograms[i][:, :sample_length]

    # Find the smallest valid spectrogram frequency height for model requirements
    max_height = max([s.shape[0] for s in (clean_spectrograms + noisy_spectrograms)])
    sample_height = 1 << max_height.bit_length() if max_height & (max_height - 1) else max_height
    sample_height = 208 # Optimized demonstration data set height

    # Pad spectrogram frequencies
    # clean_spectrograms = [np.pad(s, ((0, sample_height - s.shape[0]), (0, 0)), 'constant') for s in clean_spectrograms]
    # noisy_spectrograms = [np.pad(s, ((0, sample_height - s.shape[0]), (0, 0)), 'constant') for s in noisy_spectrograms]

    # Crop spectrogram frequencies to a height of 200
    clean_spectrograms = [s[:sample_height, :] for s in clean_spectrograms]
    noisy_spectrograms = [s[:sample_height, :] for s in noisy_spectrograms]

    # Convert the lists to numpy arrays
    clean_spectrograms = np.array(clean_spectrograms)
    noisy_spectrograms = np.array(noisy_spectrograms)

    # Create binary masks at >95% out of the clean spectrograms
    masks = np.where(clean_spectrograms > np.percentile(clean_spectrograms, 95), 1, 0)

    # Visualization
    save_spectrogram(clean_spectrograms[1], 16000, 1024, 512, "clean.png")
    save_mask(clean_spectrograms[1], "clean_spect.png")
    print(clean_spectrograms[1])
    save_mask(masks[1], "mask.png")
    print(masks[1])

    # Apply normalization to the noisy spectrograms
    for i in range(len(noisy_spectrograms)):
        min_val = np.min(noisy_spectrograms[i])
        max_val = np.max(noisy_spectrograms[i])
        noisy_spectrograms[i] = (noisy_spectrograms[i] - min_val) / (max_val - min_val)
    
    print(noisy_spectrograms[i])

    return noisy_spectrograms, np.array(masks)

def unet_model(input_shape):
    """
    Create a U-Net model for spectrogram noise reduction.

    Parameters:
    - input_shape (tuple): Shape of the input spectrograms.

    Returns:
    - keras.Model: The constructed U-Net model.
    """
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
noisy, masks = load_data("clean_speech", "noisy_speech")
width, height = noisy.shape[1], noisy.shape[2]

# Split the data into training and validation sets
noisy_train, noisy_val, masks_train, masks_val = train_test_split(
    noisy, masks, test_size = 0.2, random_state = 42
)

# Create and compile the UNet model
model = unet_model((width, height, 1))
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Debugging info for model dimensions
print(f"Noisy Train Shape: {noisy_train.shape}")
print(f"Noisy Validation Shape: {noisy_val.shape}")
print(f"Masks Train Shape: {masks_train.shape}")
print(f"Masks Validation Shape: {masks_val.shape}")

# Fit the model to the data
model.fit(
    noisy_train,
    masks_train,
    epochs = 10,
    batch_size = 8,
    validation_data = (noisy_val, masks_val)
)

# Save details
model.summary()
model.save("denoiser.keras")