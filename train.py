# # import os
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score
# # import joblib
# # from audio_utils import extract_features

# # data = []
# # labels = []
# # base_dir = "dataset"  # Ensure this folder contains 'normal/' and 'abnormal/' subfolders

# # for category in ["normal", "abnormal"]:
# #     folder = os.path.join(base_dir, category)
# #     for filename in os.listdir(folder):
# #         if filename.endswith(".wav"):
# #             path = os.path.join(folder, filename)
# #             features = extract_features(path)
# #             data.append(features)
# #             labels.append(category)

# # # Encode labels
# # le = LabelEncoder()
# # y = le.fit_transform(labels)  # normal=1, abnormal=0
# # X = np.array(data)

# # # Split and train
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# # model = RandomForestClassifier(n_estimators=100, random_state=42)
# # model.fit(X_train, y_train)

# # # Save model and label encoder
# # joblib.dump(model, "model.pkl")
# # joblib.dump(le, "label_encoder.pkl")

# # # Evaluate
# # y_pred = model.predict(X_test)
# # print("Accuracy:", accuracy_score(y_test, y_pred))
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import os
# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from audio_utils import extract_mel_spectrogram

# base_dir = "dataset"
# labels = []
# data = []

# for category in ["normal", "abnormal"]:
#     folder = os.path.join(base_dir, category)
#     for file in os.listdir(folder):
#         if file.endswith(".wav"):
#             path = os.path.join(folder, file)
#             spect = extract_mel_spectrogram(path)
#             data.append(spect)
#             labels.append(category)

# X = np.array(data)
# X = X[..., np.newaxis]  # CNN expects (samples, height, width, channels)

# le = LabelEncoder()
# y = le.fit_transform(labels)
# y = tf.keras.utils.to_categorical(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# model.save("cnn_model.h5")
# np.save("label_encoder.npy", le.classes_)

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from audio_utils import load_dataset
import os

print("Loading dataset...")
X, y = load_dataset("dataset")

print("Splitting dataset...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_val)
acc = accuracy_score(y_val, preds)
print(f"Validation Accuracy: {acc*100:.2f}%")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_audio.pkl")
print("Model saved to models/rf_audio.pkl")


