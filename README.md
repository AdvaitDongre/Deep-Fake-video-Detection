# Deepfake Detection Challenge

## Overview

This notebook provides a comprehensive approach to detecting deepfake videos using a CNN-RNN architecture. The goal is to classify videos as either REAL or FAKE. The notebook includes steps for data visualization, preprocessing, feature extraction, model training, and inference.

## Table of Contents
1. [Data Visualization](#data-visualization)
2. [Preprocessing](#preprocessing)
3. [Feature Extraction](#feature-extraction)
4. [Model Training](#model-training)
5. [Inference](#inference)
6. [Conclusion](#conclusion)

## Data Visualization

### Setup
- Imports necessary libraries such as `tensorflow`, `keras`, `cv2`, `matplotlib`, and `pandas`.
- Defines paths for training and test datasets.

### Visualizing Training and Test Data
- Displays the number of training and test samples.
- Loads metadata for the training sample videos.
- Visualizes the distribution of labels in the training set.
- Displays frames from a few fake and real videos.

### Code Snippets
```python
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

DATA_FOLDER = '../input/deepfake-detection-challenge'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'

train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T
train_sample_metadata.groupby('label')['label'].count().plot(figsize=(15, 5), kind='bar', title='Distribution of Labels in the Training Set')
plt.show()
```

## Preprocessing

### Functions
- `display_image_from_video(video_path)`: Captures and displays a frame from a video.
- `display_image_from_video_list(video_path_list)`: Captures and displays frames from a list of videos.

### Visualizing Frames
- Displays frames from a few fake and real videos.
- Displays frames from videos with the same original source.

## Feature Extraction

### Setup
- Defines parameters such as `IMG_SIZE`, `BATCH_SIZE`, `EPOCHS`, `MAX_SEQ_LENGTH`, and `NUM_FEATURES`.

### Functions
- `crop_center_square(frame)`: Crops the center square of a frame.
- `load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE))`: Loads and processes frames from a video.
- `build_feature_extractor()`: Builds a feature extractor using the InceptionV3 model pre-trained on ImageNet.

### Code Snippets
```python
from tensorflow import keras

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()
```

## Model Training

### Preparing Data
- Splits the training data into training and validation sets.
- Prepares video frames and labels for training and validation.

### Sequence Model
- Defines a sequence model consisting of GRU layers.
- Compiles the model with binary crossentropy loss and Adam optimizer.
- Trains the model using the prepared data.

### Code Snippets
```python
from tensorflow import keras

frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
x = keras.layers.GRU(8)(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Dense(8, activation="relu")(x)
output = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model([frame_features_input, mask_input], output)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

checkpoint = keras.callbacks.ModelCheckpoint('./', save_weights_only=True, save_best_only=True)
history = model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_data=([test_data[0], test_data[1]],test_labels),
        callbacks=[checkpoint],
        epochs=EPOCHS,
        batch_size=8
    )
```

## Inference

### Functions
- `prepare_single_video(frames)`: Prepares frames from a single video for prediction.
- `sequence_prediction(path)`: Predicts whether a video is REAL or FAKE.

### Visualizing Predictions
- Uses the trained model to predict the class of test videos.
- Displays the predicted class along with the video.

### Code Snippets
```python
def sequence_prediction(path):
    frames = load_video(os.path.join(DATA_FOLDER, TEST_FOLDER,path))
    frame_features, frame_mask = prepare_single_video(frames)
    return model.predict([frame_features, frame_mask])[0]

test_video = "/kaggle/input/deepfake-detection-challenge/train_sample_videos/aelfnikyqj.mp4"
if(sequence_prediction(test_video)<=0.5):
    print(f'The predicted class of the video is FAKE')
else:
    print(f'The predicted class of the video is REAL')
```

## Conclusion

The deepfake detection model, using a combination of CNN and RNN, achieves a generalization accuracy of around 80%. The approach demonstrates the ability to preprocess video data, extract meaningful features, and classify videos with reasonable accuracy.

## Requirements

- TensorFlow
- Keras
- OpenCV
- Pandas
- NumPy
- Imageio
- Matplotlib

## References

- [Keras Applications](https://keras.io/api/applications/)
- [GRU Layer in Keras](https://keras.io/api/layers/recurrent_layers/gru/)
- [TensorFlow Hub Action Recognition Tutorial](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub)

Feel free to explore and modify the code to improve the performance and adapt it to your specific use case.
