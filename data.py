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
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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

    # Reshape the spectrogram if it is 1D
    if len(spectrogram.shape) == 1:
        spectrogram = np.expand_dims(spectrogram, axis=0)

    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(save_path)
    plt.close()  # Close the figure to release resources

def save_spectrogram_visualizations(spectrograms, masks, save_folder, sr, hop_length):
    # Create the save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i in range(len(spectrograms)):
        # Display and save the spectrogram
        spectrogram_path = os.path.join(save_folder, f'spectrogram_{i}.png')
        display_spectrogram(spectrograms[i], sr, hop_length, spectrogram_path)

        # Save the binary mask as a black and white image
        mask_path = os.path.join(save_folder, f'mask_{i}.png')
        plt.imsave(mask_path, masks[i], cmap='gray')

def load_data(clean_speech_folder, noisy_speech_folder):
    db_interval = -60
    max_height = 0
    max_width = 0
    spectrograms = []
    masks = []

    # Find the dimensions of the largest image for padding
    for clean_file_name in os.listdir(clean_speech_folder):
        if clean_file_name.endswith(".wav"):
            clean_speech_path = os.path.join(clean_speech_folder, clean_file_name)
            clean_speech_spectrogram, _ = wav_to_spectrogram(clean_speech_path)
            height, width = clean_speech_spectrogram.shape
            max_height = max(max_height, height)
            max_width = max(max_width, width)

    # Update to nearest power of 2 for model compatability
    target_height = 2 ** int(np.ceil(np.log2(max_height)))
    target_width = 2 ** int(np.ceil(np.log2(max_width)))

    # Iterate through source files
    for clean_file_name in os.listdir(clean_speech_folder):
        if clean_file_name.endswith(".wav"):
            # Load the clean speech file
            clean_speech_path = os.path.join(clean_speech_folder, clean_file_name)
            clean_speech_spectrogram, _ = wav_to_spectrogram(clean_speech_path)

            # Calculate padding
            pad_height = target_height - clean_speech_spectrogram.shape[0]
            pad_width = target_width - clean_speech_spectrogram.shape[1]

            # Add padding to the clean speech spectrogram
            clean_speech_spectrogram = np.pad(clean_speech_spectrogram, ((0, pad_height), (0, pad_width)), mode='constant')

            # Create a binary mask from the clean speech file
            mask = (clean_speech_spectrogram > db_interval).astype(np.float32)

            # Normalize clean speech spectrogram to [0, 1]
            clean_speech_spectrogram = (clean_speech_spectrogram - clean_speech_spectrogram.min()) / (clean_speech_spectrogram.max() - clean_speech_spectrogram.min())

            # Iterate through files with corresponding clean speech file at all SNRs
            for noisy_file_name in os.listdir(noisy_speech_folder):
                if noisy_file_name.endswith(clean_file_name):
                    # Load the noisy speech file
                    noisy_speech_path = os.path.join(noisy_speech_folder, noisy_file_name)
                    noisy_speech_spectrogram, _ = wav_to_spectrogram(noisy_speech_path)

                    # Add padding to the noisy speech spectrogram
                    noisy_speech_spectrogram = np.pad(noisy_speech_spectrogram, ((0, pad_height), (0, pad_width)), mode='constant')

                    # Normalize noisy speech spectrogram to [0, 1]
                    noisy_speech_spectrogram = (noisy_speech_spectrogram - noisy_speech_spectrogram.min()) / (noisy_speech_spectrogram.max() - noisy_speech_spectrogram.min())

                    # Store the results in arrays
                    spectrograms.append(noisy_speech_spectrogram)
                    masks.append(mask)

    return np.array(spectrograms), np.array(masks)

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

# # Load the data
# spectrograms, masks = load_data("clean_speech", "noisy_speech")
# width, height = spectrograms.shape[1], spectrograms.shape[2]

# # Split the data into training and validation sets
# data_train, data_val, masks_train, masks_val = train_test_split(
#     spectrograms, masks, test_size = 0.2, random_state = 42
# )
# print(masks[1])

# # Create and compile the UNet model
# model = unet_model((width, height, 1))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Fit the model to the data
# model.fit(
#     data_train,
#     masks_train,
#     epochs = 10,
#     batch_size = 16,
#     validation_data = (data_val, masks_val)
# )

# model.save("denoiser.keras")

model = tf.keras.models.load_model("denoiser.keras")

# Function to apply the trained model to new data
def apply_model_to_new_data(model, file_path, target_height, target_width, db_interval=-60):
    # Convert the new audio file to a spectrogram
    spectrogram, sr, magnitude, phase = wav_to_spectrogram(file_path, save_phase=True)

    # Calculate padding
    pad_height = target_height - spectrogram.shape[0]
    pad_width = target_width - spectrogram.shape[1]

    # Add padding to the spectrogram
    spectrogram_padded = np.pad(spectrogram, ((0, pad_height), (0, pad_width)), mode='constant')

    # # Normalize the spectrogram to [0, 1]
    # spectrogram_normalized = (spectrogram_padded - spectrogram_padded.min()) / (spectrogram_padded.max() - spectrogram_padded.min())

    # Reshape the spectrogram for model input
    # input_data = np.expand_dims(spectrogram_normalized, axis=-1)
    input_data = np.expand_dims(spectrogram_padded, axis=-1)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

    # Apply the trained model
    predicted_mask = model.predict(input_data)[0, :, :, 0]

    # Post-process the predicted mask if needed (e.g., thresholding)
    threshold = 0.5
    predicted_mask_binary = (predicted_mask > threshold).astype(np.float32)

    # Optionally, convert the predicted binary mask back to a time-domain signal using saved phase information
    reconstructed_audio = spectrogram_to_wav(predicted_mask_binary, magnitude, phase, sr)

    return predicted_mask_binary, reconstructed_audio, magnitude, phase, sr

# Example usage:
file_path = "MS-SNSD/CleanSpeech_training/clnsp1.wav"
predicted_mask, reconstructed_audio, magnitude, phase, sr = apply_model_to_new_data(model, file_path, 128, 512)

# Apply the predicted binary mask to the magnitude and phase
# Ensure that predicted_mask has the same shape as magnitude
applied_mask = np.resize(predicted_mask, magnitude.shape) * magnitude

# Reconstruct the audio using the masked magnitude and original phase
reconstructed_audio = spectrogram_to_wav(applied_mask, magnitude, phase, sr)

# Save the reconstructed audio as a .wav file using scipy.io.wavfile
output_file_path = "output_reconstructed.wav"
display_spectrogram(reconstructed_audio, sr, 512, "reconstructed_spectrogram.png")
wavfile.write(output_file_path, sr, reconstructed_audio.astype(np.int16))