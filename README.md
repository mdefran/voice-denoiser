The data.py file contains the program for training the model. The input.py file can be used to predict on new data. The file should be run in an environment with TensorFlow installed, alongside CUDA and CuDNN.

The files should be placed in the same directory as an instance of MS-SNSD. By default, the data is read from "clean_speech" and "noisy_speech". If you are generating new data, be sure to update the paths "clean_speech_folder" and "noisy_speech_folder" in the data.py script, or rename and move the folders to match the existing requirements. You can adjust the "snr_intervals" in the same location.

Be sure to update the dimensions of the model input in the input.py file. These will be displayed at the start of training. Also, note the hard cap set for voice files in input.py.
