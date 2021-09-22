# https://www.tensorflow.org/tutorials/keras/text_classification
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

print(tf.__version__)

def custom_strip(data):
  lc = tf.strings.lower(data)
  sh = tf.strings.regex_replace(lc, '<br />', ' ')
  return tf.strings.regex_replace(sh, '[{}]'.format(re.escape(string.punctuation)), '')

def main(args):

  if args.download:
    print("Downloading training data ..")
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='')
    
    dataset_dir = os.path.join(os.path.dirname(dataset), 'data/aclImdb')
    train_dir = os.path.join(dataset_dir, 'train')
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

  """
  When running a machine learning experiment, it is a best practice to divide your 
  dataset into three splits: train, validation, and test.

  The IMDB dataset has already been divided into train and test, but it lacks a validation set.
  Let's create a validation set using an 80:20 split of the training data by using the 
  validation_split argument below.
  """

  batch_size = 32
  seed = 42
  raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, # 80/20
    subset='training', 
    seed=seed)

  for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
      print("Review", text_batch.numpy()[i])
      print("Label", label_batch.numpy()[i])

  # Next, you will create a validation and test dataset. You will use the remaining 5,000 reviews from the training set for validation.
  raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)
  
  # The test dataset already exists at 'aclImdb/test'
  raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/test', batch_size=batch_size)

  max_features = 10000
  sequence_length = 250

  # Build a vectorization layer to standardize, tokenize, and vectorize our data
  vectorize_layer = TextVectorization(
    standardize=custom_strip,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length = sequence_length)
  
  # Make a text-only dataset (without labels), then call adapt
  train_text = raw_train_ds.map(lambda x, y: x)
  vectorize_layer.adapt(train_text)

  def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

  # retrieve one batch (of 32 reviews and labels) from the dataset
  text_batch, label_batch = next(iter(raw_train_ds))
  first_review, first_label = text_batch[0], label_batch[0]
  print("Review", first_review)
  print("Label", raw_train_ds.class_names[first_label])
  print("Vectorized review", vectorize_text(first_review, first_label))
  print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
  print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
  print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

  # Now apply the vectorization layer to all the datasets
  train_ds = raw_train_ds.map(vectorize_text)
  val_ds   = raw_val_ds.map(vectorize_text)
  test_ds  = raw_test_ds.map(vectorize_text)

  AUTOTUNE = tf.data.AUTOTUNE
  train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

  # Now, let's create the model
  embedding_dim = 16
  model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])

  model.summary()

  model.compile(
    loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

  # Train the model
  history = model.fit(train_ds, validation_data=val_ds, epochs=10)

  # Evaluate the model
  loss, accuracy = model.evaluate(test_ds)
  print("Loss: {}, Accuracy: {}".format(loss, accuracy))

  # Print the history of training
  history_dict = history.history
  acc = history_dict['binary_accuracy']
  val_acc = history_dict['val_binary_accuracy']
  loss = history_dict['loss']
  val_loss = history_dict['val_loss']

  epochs = range(1, len(acc) + 1)

  # "bo" is for "blue dot"
  plt.plot(epochs, loss, 'bo', label='Training loss')
  # b is for "solid blue line"
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  plt.show()

  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')

  plt.show()

  # Export the model
  # This model exports the vectorization steps also!
  # "Including the text preprocessing logic inside your model enables you to export a model 
  # for production that simplifies deployment, and reduces the potential for train/test skew."
  export_model = tf.keras.Sequential([vectorize_layer, model, layers.Activation('sigmoid')])
  export_model.compile(
      loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
  )

  # Test it with `raw_test_ds`, which yields raw strings
  loss, accuracy = export_model.evaluate(raw_test_ds)
  print(accuracy)

  # Get some predictions
  examples = [
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible..."
  ]

  print(export_model.predict(examples))

def setup_argparse():
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--download', type=bool, help='Download the dataset')
  
  return parser.parse_args()


if __name__ == "__main__":
  main(setup_argparse())
