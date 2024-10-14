from jax import random, nn
from jax_tqdm import scan_tqdm
from tqdm import tqdm
import jax.numpy as jnp
import jax
from optax import adam, apply_updates, softmax_cross_entropy_with_integer_labels


def classifier():
    def init_model(rng, layers, kernels):
        stride = 1

        def conv_step(input, kernel):
            z = jax.lax.conv(input, kernel, (stride, stride), padding="SAME")
            return z

        def dense_step(input, weight, bias):
            return jnp.dot(input, weight) + bias

        @jax.jit
        def forward(params, input):
            batch_size = input.shape[0]

            # convolution nchw
            for kernel in params["conv"]:
                z = conv_step(input, kernel)
                input = jax.nn.relu(z)

            z = input.reshape(batch_size, -1)

            # dense
            for w, b in params["dense"][:-1]:
                z = dense_step(z, w, b)
                z = jax.nn.relu(z)
            w, b = params["dense"][-1]
            z = dense_step(z, w, b)
            return z

        def init_dense(rng, layers):
            def init_layer(i, o, key):
                w = random.normal(key, (i, o)) / jnp.sqrt(i)
                b = jnp.zeros(o)
                return (w, b)

            keys = random.split(rng, len(layers))
            params = []

            for i, o, key in zip(layers[:-1], layers[1:], keys):
                params.append(init_layer(i, o, key))
            # params = init_layer(layers[:-1], layers[1:], keys)
            return params

        def init_conv(rng, kernels):
            def init_kernel(channel_output, channel_input, kernel_size, key):
                kernel = random.normal(
                    key, (channel_output, channel_input, kernel_size, kernel_size)
                ) / jnp.sqrt(channel_input * kernel_size * kernel_size)
                return kernel

            keys = random.split(rng, len(kernels) - 1)
            # conv_params = init_kernel(kernels[:-1], kernels[1:], 3, keys)
            conv_params = []
            for (o, i, size), key in zip(kernels, keys):
                conv_params.append(init_kernel(o, i, size, key))
            return conv_params

        # model initialisation
        rngs = random.split(rng, 3)
        dense_params = init_dense(rngs[0], layers)
        conv_params = init_conv(rngs[1], kernels)
        params = {"conv": conv_params, "dense": dense_params}
        return params, forward

    def predict(forward, params, images):
        probs = nn.softmax(forward(params, images))
        return jnp.argmax(probs, axis=1)

    def train_model(rng, params, forward, images, labels, cfg):
        batch_size, lr, n_epochs = cfg

        def shuffle_data(key, images, labels):
            # randomly shuffle the data
            indices = random.permutation(key, jnp.arange(len(images)))

            # divide the data in batches
            images = images[indices]
            labels = labels[indices]

            images = images.reshape(-1, batch_size, 1, 28, 28)
            labels = labels.reshape(-1, batch_size)
            return zip(images, labels), len(images)

        @jax.jit
        def update_fn(params, batch, opt_state):
            @jax.value_and_grad
            def loss_fn(params, batch):
                images, labels = batch
                logits = forward(params, images)
                loss = softmax_cross_entropy_with_integer_labels(logits, labels)
                return jnp.mean(loss)

            loss, grads = loss_fn(params, batch)
            updates, opt_state = opt.update(grads, opt_state)
            params = apply_updates(params, updates)
            return params, loss, opt_state

        # training process
        opt = adam(lr)
        opt_state = opt.init(params)
        losses = []

        for epoch in range(n_epochs):
            print(epoch)
            rng, key = random.split(rng)
            shuffled_data, length = shuffle_data(key, images, labels)

            for img, lbl in (pbar := tqdm(shuffled_data, total=length)):
                params, loss, opt_state = update_fn(params, (img, lbl), opt_state)
                pbar.set_description(f"Loss: {loss:.3f}")
                losses.append(loss)
        return params, losses

    return init_model, train_model, predict
