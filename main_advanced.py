import jax
import jax.numpy as jnp
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from classifier_advanced import classifier

# from classifier_naive import classifier
import tensorflow_datasets as tfds

from loader import load_params, save_params


mnist = tfds.load("mnist", split="train")
images = (
    jnp.array([x["image"] for x in tfds.as_numpy(mnist)]).astype(jnp.float32) / 255.0
)
images = images.reshape(-1, 1, 28, 28)
labels = jnp.array([x["label"] for x in tfds.as_numpy(mnist)])

train_images = images[:16000]
train_labels = labels[:16000]

test_images = images[16000:]
test_labels = labels[16000:]

rng = jax.random.PRNGKey(0)
conv_output_size = 128
layers = [28 * 28, 64, 128, 10]
kernels = [[32, 1, 3], [64, 32, 3], [1, 64, 3], [1, conv_output_size, 3]]
# kernels = [[32, 1, 3], [64, 32, 3], [128, 64, 3], [1, conv_output_size, 3]]
lr = 0.001
n_epochs = 10

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit()

    arg = sys.argv[1]
    init_classifier, train_classifier, predict = classifier()
    params, forward, forward_vmapped = init_classifier(rng, layers, kernels)

    if arg == "train":
        # train classifier
        params, losses = train_classifier(
            rng, params, forward_vmapped, train_images, train_labels, (lr, n_epochs)
        )
        save_params("params.npz", params)

    elif arg == "test":
        params = load_params("params.npz")
        print("predicting...")
        y = predict(forward_vmapped, params, test_images)

        # accuracy
        print(f"Accuracy: {(y == test_labels).astype(jnp.int32).mean()}")

    elif arg == "gif":
        images = [[], [], [], [], []]
        n_saves = 10
        for epoch in range(n_saves):
            print(f"Loading params_{epoch}")
            params = load_params(f"params_{epoch}.npz")
            _, activations = predict(
                forward, params, test_images[0], get_activation=True
            )
            images[0].append(activations[2].reshape(28, 28, 1))  # layer finale
            images[1].append(activations[0][0].reshape(28, 28, 1))
            images[2].append(activations[0][5].reshape(28, 28, 1))
            images[3].append(activations[1][0].reshape(28, 28, 1))
            images[4].append(activations[0][2].reshape(28, 28, 1))

        for i in range(0, 5):
            fig, ax = plt.subplots()
            im = ax.imshow(images[i][0], cmap="gray", animated=True)

            def update(j):
                im.set_array(images[i][j])
                return (im,)

            animation_fig = animation.FuncAnimation(
                fig,
                update,
                frames=len(images[i]),
                interval=200,
                blit=True,
                repeat_delay=10,
            )
            animation_fig.save(f"animated_{i}.gif")

    else:
        print("Invalid argument")
        sys.exit()
