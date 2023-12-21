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

The Microsoft Scalable Noisy Speech Dataset (MS-SNSD) is a comprehensive audio dataset that features recordings sampled at 16 kHz and stored as .wav files. It includes short clips of various researchers and practitioners reading excerpts from literature, ensuring a diverse representation in terms of gender and accents. The dataset also has a range of background noises from different sources, including air conditioners, babble, copy machines, typing, and vacuum cleaners. I merged together clips of each type to simulate speech with natural background noise, logging the corresponding clean speech files to use as binary masks. I then fed this data into a U-Net model.

