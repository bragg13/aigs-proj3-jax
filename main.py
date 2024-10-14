import jax
import jax.numpy as jnp
import sys
import matplotlib.pyplot as plt
import os

# from classifier_advanced import classifier
from classifier_naive import classifier
import tensorflow_datasets as tfds

from loader import load_params, save_params


mnist = tfds.load("mnist", split="train")
images = (
    jnp.array([x["image"] for x in tfds.as_numpy(mnist)]).astype(jnp.float32) / 255.0
)
images = images.reshape(-1, 1, 28, 28)
labels = jnp.array([x["label"] for x in tfds.as_numpy(mnist)])

train_images = images[:30000]
train_labels = labels[:30000]

test_images = images[30000:]
test_labels = labels[30000:]

rng = jax.random.PRNGKey(0)
conv_output_size = 128
layers = [28 * 28 * conv_output_size, 64, 128, 10]  #
kernels = [[32, 1, 3], [64, 32, 3], [128, 64, 3], [32, conv_output_size, 3]]
batch_size = 30  # divisibile per 60000
lr = 0.001
n_epochs = 10

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit()

    arg = sys.argv[1]
    init_classifier, train_classifier, predict = classifier()
    params, forward = init_classifier(rng, layers, kernels)

    if arg == "train":
        # train classifier
        params, losses = train_classifier(
            rng, params, forward, train_images, train_labels, (batch_size, lr, n_epochs)
        )
        save_params("params.npz", params)

    elif arg == "test":
        params = load_params("params.npz")
        print(f'{params['conv'][0][0]}')
        print("predicting...")
        y = predict(forward, params, test_images.reshape(-1, 1, 28, 28))

        # accuracy
        print(f"Accuracy: {(y == test_labels).astype(jnp.int32).mean()}")

    else:
        print("Invalid argument")
        sys.exit()
