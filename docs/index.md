# Speech Denoiser
### Matthew DeFranceschi  
mmd22@njit.edu

New Jersey Institute of Technology  
CS / IS 485: Special Topic: Machine Listening  
Professor: Mark Cartwright

In today's world, both personal and professional lifestyles are increasingly dependent on digital communication, predominantly through voice calls and video meetings. This trend has been further accelerated by the global shift towards remote work, emphasizing the importance of such virtual interactions. The cornerstone of these virtual meetings is the clarity of audio, which is paramount for effective communication. However, there is a notable challenge in this aspect: high-quality audio equipment, crucial for clear communication, often comes with a hefty price tag, making it an unreasonable investment for every end user. Additionally, the variety of settings in which these meetings take place introduces a range of background noises, each presenting its challenges to the clarity and quality of the communication.

Software can solve this issue by using machine learning models to cancel out background noise. There are already existing products that accomplish this, built into Zoom, Team, and Discord through Krisp. I recreated a form of this software for files provided by an end user.

My approach followed this general outline:
1. Generate noisy data by merging together background noise and clean speech clips
2. Create binary masks out of the corresponding clean speech for each noisy data file
3. Train a U-Net model on the data
4. Predict on new data input by the user

The Microsoft Scalable Noisy Speech Dataset (MS-SNSD) is a comprehensive audio dataset that features recordings sampled at 16 kHz and stored as .wav files. It includes short clips of various researchers and practitioners reading excerpts from literature, ensuring a diverse representation in terms of gender and accents. The dataset also has a range of background noises from different sources, including air conditioners, babble, copy machines, typing, and vacuum cleaners. I merged together clips of each type to simulate speech with natural background noise, logging the corresponding clean speech files to use as binary masks. I then fed this data into a U-Net model, using the masks as labels.

When making predictions on new data using the model, I ensured that I applied the same transformations to the input data as I did when training the model. I used the result to determine which values should be entirely omitted from the resulting spectrogram. Then, I explored a few options for reconstructing the STFT and resulting audio file from the prediction. One option was to use the saved phase information directly. Another was to iteratively improve a phase information estimate to match the magnitude spectrogram. Finally, a trained neural vocoder can predict these values for reconstruction.

In terms of evaluation, most speech denoising competitions use subjective evalution from a large number of volunteers. However, I do not have access to this platform. Luckily, Microsoft developed DNSMOS P.835, which is a machine learning model trained on past competitions. This is a promising method of evaluation. Unfortunately, I do not have access to it. Regardless, SDR can be used to evaluate the performance.

I faced a number of challenges and setbacks while completing this project. I initially worked with Mel spectrograms, which was detrimental to my model as it discards important information. Furthermore, it does not work effectively with mask thresholds. However, without log compression, I struggled to find a way to reduce the dimensionality without losing frequency or temporal information. This effectively made the data set huge and difficult to handle given my hardware and memory limitations. Meanwhile, I encountered overfitting and misleading accuracy values which simply suppressed the entire audio clip. Finally, I encountered some negative interactions between binary cross entropy and the unbalanced nature of the data.
