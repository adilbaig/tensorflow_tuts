# https://www.tensorflow.org/tutorials/keras/classification

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

CLASS_NAMES = [
    'T-Shirt',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle Boot',
]

print(tf.__version__)


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(CLASS_NAMES[predicted_label],
                                         100*np.max(predictions_array),
                                         CLASS_NAMES[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def setup_argparse():
    import argparse

    parser = argparse.ArgumentParser(
        description='Run Image classification. Opetionally export and import models')
    parser.add_argument('--import-model', type=str,
                        help='Import a Keras model from this path.')
    parser.add_argument('--export', type=str,
                        help='Export a trained model to this path.')

    return parser.parse_args()


def main(args):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print("train_images.shape", train_images.shape)
    print("test_labels", test_labels)
    print("test_images.shape", test_images.shape)

    # Step1: Preprocess the images before training
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    if not args.import_model:
      # Step2: Build the model
      model = tf.keras.Sequential([
          # Transform from 2D to 1D array
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(128, activation='relu'),  # 128 Neurons/Nodes
          tf.keras.layers.Dense(10),  # 10 Logits
      ])

      model.compile(
          optimizer='adam',
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          metrics=['accuracy']
      )

      # Step3: Train the neural network
      model.fit(train_images, train_labels, epochs=10)
    else:
      print("Importing trained model from", args.import_model)
      model = tf.keras.models.load_model(args.import_model)

    if args.export:
      print("Saving model to", args.export)
      model.save(args.export)
    
    # Step4: Evaluate accuracy
    # test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:{1}, nTest loss:{0}'.format(*model.evaluate(test_images, test_labels, verbose=2)))

    # Step5: DONE TRAINING! Start making predictions on test images
    # Softmax converts logitcs to probabilities
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)  # predictions work in batches

    # Pick a random result and show the prediction
    i = random.randint(0, len(predictions))
    plot_image(i, predictions[i], test_labels, test_images)
    plt.show()


if __name__ == "__main__":
    args = setup_argparse()
    main(args)
