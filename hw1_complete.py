#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import layers, Sequential

# Helper libraries
import numpy as np
import os


print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


def build_model1():
  """4-layer fully-connected: Flatten + 3 Dense(128, leaky_relu) + Dense(10)."""
  model = Sequential([
      layers.Flatten(input_shape=(32, 32, 3)),
      layers.Dense(128, activation='leaky_relu'),
      layers.Dense(128, activation='leaky_relu'),
      layers.Dense(128, activation='leaky_relu'),
      layers.Dense(10),
  ])
  return model


def build_model2():
  """CNN: Conv 32 stride2 -> BN -> Conv 64 stride2 -> BN -> 4x(Conv 128 -> BN) -> Flatten -> Dense(10)."""
  model = Sequential([
      layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu', input_shape=(32, 32, 3)),
      layers.BatchNormalization(),
      layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Flatten(),
      layers.Dense(10),
  ])
  return model


def build_model3():
  """Same as model2 but all conv layers are SeparableConv2D."""
  model = Sequential([
      layers.SeparableConv2D(32, (3, 3), strides=2, padding='same', activation='relu', input_shape=(32, 32, 3)),
      layers.BatchNormalization(),
      layers.SeparableConv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.Flatten(),
      layers.Dense(10),
  ])
  return model


def build_model50k():
  """Best model with no more than 50,000 parameters (tuned for validation accuracy)."""
  model = Sequential([
      layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu', input_shape=(32, 32, 3)),
      layers.BatchNormalization(),
      layers.SeparableConv2D(64, (3, 3), strides=2, padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
      layers.BatchNormalization(),
      layers.GlobalAveragePooling2D(),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.3),
      layers.Dense(10),
  ])
  return model


# no training or dataset construction should happen above this line
# also, be careful not to unindent below here, or the code be executed on import
if __name__ == '__main__':

  ########################################
  ## Load the CIFAR10 data set and split train/val
  ########################################
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  train_labels = train_labels.squeeze()
  test_labels = test_labels.squeeze()

  # Normalize to [0, 1]
  train_images = train_images.astype(np.float32) / 255.0
  test_images = test_images.astype(np.float32) / 255.0

  # Use last 5000 training samples as validation
  n_val = 5000
  val_images = train_images[-n_val:]
  val_labels = train_labels[-n_val:]
  train_images = train_images[:-n_val]
  train_labels = train_labels[:-n_val]

  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  epochs = 30

  ########################################
  ## Build and train model 1
  ########################################
  model1 = build_model1()
  model1.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
  model1.summary()
  model1.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels), verbose=1)
  train_acc1 = model1.evaluate(train_images, train_labels, verbose=0)[1]
  val_acc1 = model1.evaluate(val_images, val_labels, verbose=0)[1]
  test_acc1 = model1.evaluate(test_images, test_labels, verbose=0)[1]
  print(f"Model1 - Train acc: {train_acc1:.4f}, Val acc: {val_acc1:.4f}, Test acc: {test_acc1:.4f}")

  ########################################
  ## Build, compile, and train model 2
  ########################################
  model2 = build_model2()
  model2.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
  model2.summary()
  model2.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels), verbose=1)
  train_acc2 = model2.evaluate(train_images, train_labels, verbose=0)[1]
  val_acc2 = model2.evaluate(val_images, val_labels, verbose=0)[1]
  test_acc2 = model2.evaluate(test_images, test_labels, verbose=0)[1]
  print(f"Model2 - Train acc: {train_acc2:.4f}, Val acc: {val_acc2:.4f}, Test acc: {test_acc2:.4f}")

  ########################################
  ## Build, compile, and train model 3 (depthwise-separable)
  ########################################
  model3 = build_model3()
  model3.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
  model3.summary()
  model3.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels), verbose=1)
  train_acc3 = model3.evaluate(train_images, train_labels, verbose=0)[1]
  val_acc3 = model3.evaluate(val_images, val_labels, verbose=0)[1]
  test_acc3 = model3.evaluate(test_images, test_labels, verbose=0)[1]
  print(f"Model3 - Train acc: {train_acc3:.4f}, Val acc: {val_acc3:.4f}, Test acc: {test_acc3:.4f}")

  ## Test image inference
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  # Run inference on aeroplane.jpg with model2 and model3
  test_image_path = './aeroplane.jpg'
  if os.path.isfile(test_image_path):
    # Load image with keras load_img: target_size=(32,32) crops/rescales to 32x32
    test_img = np.array(keras.utils.load_img(
        test_image_path,
        grayscale=False,
        color_mode='rgb',
        target_size=(32, 32)))
    test_img = (test_img.astype(np.float32) / 255.0)[np.newaxis, ...]   # normalize and add batch dim

    # Inference from model2
    logits2 = model2.predict(test_img, verbose=0)
    pred2 = np.argmax(logits2[0])
    print(f"[Model2] {test_image_path} -> predicted: {class_names[pred2]}")

    # Inference from model3
    logits3 = model3.predict(test_img, verbose=0)
    pred3 = np.argmax(logits3[0])
    print(f"[Model3] {test_image_path} -> predicted: {class_names[pred3]}")

    # Expected class for aeroplane.jpg is 'airplane'
    expected_class = 'airplane'
    correct2 = (class_names[pred2] == expected_class)
    correct3 = (class_names[pred3] == expected_class)
    print(f"Does it correctly label the picture? (expected: {expected_class})")
    print(f"  Model2: {'Yes' if correct2 else 'No'}")
    print(f"  Model3: {'Yes' if correct3 else 'No'}")
  else:
    print(f"Test image not found: {test_image_path}")

  # Optional: classify any test_image_<classname>.<ext> files (e.g. test_image_airplane.jpg)
  for name in class_names:
    for ext in ['png', 'jpg']:
      path = f'./test_image_{name}.{ext}'
      if os.path.isfile(path):
        test_img = np.array(keras.utils.load_img(path, grayscale=False, color_mode='rgb', target_size=(32, 32)))
        test_img = (test_img.astype(np.float32) / 255.0)[np.newaxis, ...]
        for model_name, model in [('model2', model2), ('model3', model3)]:
          logits = model.predict(test_img, verbose=0)
          pred = np.argmax(logits[0])
          correct = (class_names[pred] == name)
          print(f"[{model_name}] {path} -> predicted: {class_names[pred]} (filename says: {name}). Does it correctly label the picture? {'Yes' if correct else 'No'}")
        break

  ## Best model (<=50k params), train and save
  model50k = build_model50k()
  model50k.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=loss_fn, metrics=['accuracy'])
  model50k.summary()
  print(f"Model50k params: {model50k.count_params()}")
  model50k.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels), verbose=1)
  model50k.save("best_model.h5")
  test_acc50k = model50k.evaluate(test_images, test_labels, verbose=0)[1]
  print(f"Best model test accuracy: {test_acc50k:.4f}")
