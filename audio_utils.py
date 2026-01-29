import librosa
import numpy as np
import os
import glob
import sounddevice as sd
import scipy.io.wavfile as wav

# Extract features (flattened mel spectrogram)
def extract_features(file_path, sr=22050, n_mels=64, n_fft=1024, hop_length=512):
    y, sr = librosa.load(file_path, sr=sr, duration=6)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr,
                                        n_fft=n_fft,
                                        hop_length=hop_length,
                                        n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db.flatten()  # flatten to 1D feature vector

# Build dataset
def load_dataset(root_dir):
    X, y = [], []
    normal_files = glob.glob(os.path.join(root_dir, "normal", "*.wav"))
    abnormal_files = glob.glob(os.path.join(root_dir, "abnormal", "*.wav"))

    for f in normal_files:
        X.append(extract_features(f))
        y.append(0)
    for f in abnormal_files:
        X.append(extract_features(f))
        y.append(1)

    return np.array(X), np.array(y)

# Record audio (6 seconds)
def record_audio(filename="recorded.wav", duration=6, sr=22050):
    print(f"Recording {duration} seconds...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    wav.write(filename, sr, (recording * 32767).astype(np.int16))
    print(f"Saved {filename}")
    return filename



# # import librosa
# # import numpy as np
# # import sounddevice as sd
# # from scipy.io.wavfile import write
# # import tempfile

# # # Extract MFCC features from a WAV file
# # def extract_features(file_path, max_pad_len=174):
# #     audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 
# #     audio = audio / np.max(np.abs(audio))  # Normalize
# #     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
# #     if mfccs.shape[1] < max_pad_len:
# #         pad_width = max_pad_len - mfccs.shape[1]
# #         mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
# #     else:
# #         mfccs = mfccs[:, :max_pad_len]
# #     return mfccs.flatten()

# # # Record audio from mic and save as temp file
# # def record_audio(duration=5, fs=22050):  # You can increase duration here
# #     print("Recording...")
# #     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
# #     sd.wait()
# #     print("Recording Completed")
# #     temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
# #     write(temp_wav.name, fs, audio)
# #     return temp_wav.name

# # import librosa
# # import numpy as np
# # import sounddevice as sd
# # from scipy.io.wavfile import write
# # import tempfile

# # def record_audio(duration=10, fs=22050):
# #     print("Recording...")
# #     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
# #     sd.wait()
# #     print("Recording complete")
# #     temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
# #     write(temp_wav.name, fs, audio)
# #     return temp_wav.name

# # def extract_mel_spectrogram(file_path, max_pad_len=128):
# #     audio, sr = librosa.load(file_path, sr=22050)
# #     mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
# #     mel_db = librosa.power_to_db(mel, ref=np.max)
# #     if mel_db.shape[1] < max_pad_len:
# #         pad_width = max_pad_len - mel_db.shape[1]
# #         mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
# #     else:
# #         mel_db = mel_db[:, :max_pad_len]
# #     return mel_db

# import librosa
# import numpy as np
# import sounddevice as sd
# from scipy.io.wavfile import write
# import tempfile

# def record_audio(duration=6, fs=22050):  # Changed duration to 6 seconds
#     print("Recording...")
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
#     sd.wait()
#     print("Recording complete")
#     temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
#     write(temp_wav.name, fs, audio)
#     return temp_wav.name

# def extract_mel_spectrogram(file_path, max_pad_len=128):
#     audio, sr = librosa.load(file_path, sr=22050)
#     mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
#     mel_db = librosa.power_to_db(mel, ref=np.max)
#     if mel_db.shape[1] < max_pad_len:
#         pad_width = max_pad_len - mel_db.shape[1]
#         mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
#     else:
#         mel_db = mel_db[:, :max_pad_len]
#     return mel_db

