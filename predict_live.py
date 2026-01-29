from audio_utils import extract_features
import sounddevice as sd
import joblib
import scipy.io.wavfile as wav
from telegram_utils import send_telegram_message

DURATION = 6  # seconds
SAMPLE_RATE = 22050

print("Loading trained model....")
model = joblib.load("models/rf_audio.pkl")

print("Recording Audio (1NMP411)...")
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
sd.wait()
wav.write("recorded.wav", SAMPLE_RATE, recording)

print("Recording finished. Saved as recorded.wav")

features = extract_features("recorded.wav").reshape(1, -1)
prediction = model.predict(features)[0]

# Map numeric prediction to label
if prediction == 0:
    label = " Engine Normal "
else:
    label = " Engine Abnormal condition "

msg = f"Model Prediction: {label}"
print(msg)

# Send Telegram notification
send_telegram_message(msg)





# # # import joblib
# # # from audio_utils import extract_features, record_audio

# # # def predict_live_audio(model_path="model.pkl", label_path="label_encoder.pkl"):
# # #     model = joblib.load(model_path)
# # #     le = joblib.load(label_path)

# # #     audio_file = record_audio(duration=5)
# # #     features = extract_features(audio_file).reshape(1, -1)
# # #     prediction = model.predict(features)[0]
# # #     label = le.inverse_transform([prediction])[0]

# # #     if label == "normal":
# # #         print("✅ Machinery is in normal condition.")
# # #     else:
# # #         print("❌ Machine is in fault state!")

# # # predict_live_audio()

# # import os
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # import tensorflow as tf
# # import numpy as np

# # from audio_utils import record_audio, extract_mel_spectrogram

# # def predict_live_audio(model_path="cnn_model.h5", label_path="label_encoder.npy"):
# #     model = tf.keras.models.load_model(model_path)
# #     labels = np.load(label_path)

# #     recorded_path = record_audio()
# #     spect = extract_mel_spectrogram(recorded_path)
# #     spect = spect[np.newaxis, ..., np.newaxis]  # shape (1, 128, 128, 1)

# #     prediction = model.predict(spect)[0]
# #     predicted_label = labels[np.argmax(prediction)]
    
# #     print(f"Prediction: {predicted_label}")
# #     if predicted_label == "normal":
# #         print("✅ Machinery in normal condition")
# #     else:
# #         print("⚠️ Machine is in fault state")

# # predict_live_audio()

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import sys
# sys.stdout.reconfigure(encoding='utf-8')  # For emoji support

# import tensorflow as tf
# import numpy as np

# from audio_utils import record_audio, extract_mel_spectrogram

# def predict_live_audio(model_path="cnn_model.h5", label_path="label_encoder.npy"):
#     model = tf.keras.models.load_model(model_path)
#     labels = np.load(label_path)

#     recorded_path = record_audio(duration=6)  # Record for 6 seconds
#     spect = extract_mel_spectrogram(recorded_path)
#     spect = spect[np.newaxis, ..., np.newaxis]  # shape (1, 128, 128, 1)

#     prediction = model.predict(spect)[0]
#     predicted_label = labels[np.argmax(prediction)]
    
#     print(f"Prediction: {predicted_label}")
#     if predicted_label == "normal":
#         print("✅ Machinery in normal condition")
#     else:
#         print("⚠️ Machine is in fault state")

# predict_live_audio()

# import joblib
# from audio_utils import extract_features, record_audio

# def predict(file_path, model):
#     feat = extract_features(file_path)
#     feat = feat.reshape(1, -1)  # reshape for sklearn
#     pred = model.predict(feat)[0]
#     return "Normal" if pred == 0 else "Abnormal"

# print("Loading trained model...")
# model = joblib.load("models/rf_audio.pkl")
# print("Model loaded.")

# filename = record_audio()
# result = predict(filename, model)
# print(f"Prediction: {result}")

# import joblib
# from audio_utils import extract_features, record_audio
# from telegram_utils import send_sms   # ✅ Import your SMS function

# ================== PREDICT FUNCTION ==================









# def predict(file_path, model):
#     feat = extract_features(file_path)
#     feat = feat.reshape(1, -1)  # reshape for sklearn
#     pred = model.predict(feat)[0]
#     return "Normal" if pred == 0 else "Abnormal"

# # ================== MAIN FLOW ==================
# print("Loading trained model...")
# model = joblib.load("models/rf_audio.pkl")
# print("Model loaded.")

# filename = record_audio()  # record live audio
# result = predict(filename, model)

# print(f"Prediction: {result}")

# # ✅ Send SMS Alert (using your sms_utils.py)
# send_sms(f"Audio Prediction Result: {result}")



# from audio_utils import extract_features
# import sounddevice as sd
# import joblib
# import scipy.io.wavfile as wav
# from telegram_utils import send_telegram_message

# DURATION = 6  # seconds
# SAMPLE_RATE = 22050

# print("Loading trained model...")
# model = joblib.load("models/rf_audio.pkl")

# print("Recording...")
# recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
# sd.wait()
# wav.write("recorded.wav", SAMPLE_RATE, recording)

# print("Recording finished. Saved as recorded.wav")

# features = extract_features("recorded.wav").reshape(1, -1)
# prediction = model.predict(features)[0]

# msg = f"Prediction: {prediction}"
# print(msg)

# # Send Telegram notification
# send_telegram_message(msg)





