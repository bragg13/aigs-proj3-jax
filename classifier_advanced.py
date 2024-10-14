from jax import random, nn, vmap
import jax.numpy as jnp
from jax.lax import scan
import jax
from optax import adam, apply_updates, softmax_cross_entropy_with_integer_labels
from loader import save_params


def classifier():
    def init_model(rng, layers, kernels):
        stride = 1

        def conv_step(input, kernel):
            z = jax.lax.conv_general_dilated(
                input,
                kernel,
                (stride, stride),
                padding="SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )
            return z

        def dense_step(input, weight, bias):
            return jnp.dot(input, weight) + bias

        @jax.jit
        def forward(params, input):
            # convolution nchw
            kernel_activations = []

            for kernel in params["conv"]:
                z = conv_step(input[None, ...], kernel)  # Add batch dimension
                input = jax.nn.relu(z[0])  # Remove batch dimension
                kernel_activations.append(input)

            z = input.reshape(-1)

            # dense
            for w, b in params["dense"][:-1]:
                z = dense_step(z, w, b)
                z = jax.nn.relu(z)
            w, b = params["dense"][-1]
            z = dense_step(z, w, b)
            return z, kernel_activations

        def init_dense(rng, layers):
            def init_layer(i, o, key):
                w = random.normal(key, (i, o)) / jnp.sqrt(i)
                b = jnp.zeros(o)
                return (w, b)

            keys = random.split(rng, len(layers))
            params = []

            for i, o, key in zip(layers[:-1], layers[1:], keys):
                params.append(init_layer(i, o, key))
            return params

        def init_conv(rng, kernels):
            def init_kernel(channel_output, channel_input, kernel_size, key):
                kernel = random.normal(
                    key, (channel_output, channel_input, kernel_size, kernel_size)
                ) / jnp.sqrt(channel_input * kernel_size * kernel_size)
                return kernel

            keys = random.split(rng, len(kernels) - 1)
            conv_params = []
            for (o, i, size), key in zip(kernels, keys):
                conv_params.append(init_kernel(o, i, size, key))
            return conv_params

        # model initialisation
        rngs = random.split(rng, 3)
        dense_params = init_dense(rngs[0], layers)
        conv_params = init_conv(rngs[1], kernels)
        params = {"conv": conv_params, "dense": dense_params}
        forward_vmapped = vmap(forward, in_axes=(None, 0))
        return params, forward, forward_vmapped

    def predict(forward, params, images, get_activation=False):
        z, activation = forward(params, images)
        probs = nn.softmax(z)
        if get_activation:
            # TODO this is SO BAD - get_activation also means it's for the gif
            # because i only use it in that case
            return jnp.argmax(probs, axis=0), activation
        return jnp.argmax(probs, axis=1)

    def train_model(rng, params, forward, images, labels, cfg):
        lr, n_epochs = cfg

        def shuffle_data(key, images, labels):
            # randomly shuffle the data
            rng, key = random.split(key)
            indices = random.permutation(rng, jnp.arange(len(images)))
            return images[indices], labels[indices]

        @jax.jit
        def update_fn(params, batch, opt_state):
            def loss_fn(params, batch):
                images, labels = batch
                logits, _ = forward(params, images)
                loss = softmax_cross_entropy_with_integer_labels(logits, labels)
                return jnp.mean(loss)

            loss, grads = jax.value_and_grad(loss_fn)(params, batch)
            updates, opt_state = opt.update(grads, opt_state)
            params = apply_updates(params, updates)
            return params, loss, opt_state

        @jax.jit
        def update_fn_scan(carry, batch):
            params, opt_state = carry

            def loss_fn(params, batch):
                images, labels = batch
                logits, activation = forward(params, images)
                loss = softmax_cross_entropy_with_integer_labels(logits, labels)
                return jnp.mean(loss)

            loss, grads = jax.value_and_grad(loss_fn)(params, batch)
            updates, opt_state = opt.update(grads, opt_state)
            params = apply_updates(params, updates)
            return (params, opt_state), loss

        # @jax.jit
        # def get_training_step_fn(img, lbl):
        #     @scan_tqdm(n_epochs, print_rate=10)
        #     def training_step(state, batch):
        #         params, opt_state = state
        #         images, labels = batch
        #         shuffled_images, shuffled_labels = shuffle_data(key, img, lbl)
        #         batch_size = 64
        #         n_samples = len(shuffled_images)
        #         n_batches = n_samples // batch_size
        #         batches = (
        #             shuffled_images[: n_batches * batch_size].reshape(
        #                 n_batches, batch_size, 1, 28, 28
        #             ),
        #             shuffled_labels[: n_batches * batch_size].reshape(
        #                 n_batches, batch_size
        #             ),
        #         )
        #
        #         params, loss, opt_state = update_fn(params, batches, opt_state)
        #         return (params, opt_state), loss
        #
        #     return training_step

        # training process
        opt = adam(lr)
        opt_state = opt.init(params)
        losses = []
        batch_size = 64

        for epoch in range(n_epochs):
            print(f"training epoch {epoch}")
            rng, key = random.split(rng)
            shuffled_images, shuffled_labels = shuffle_data(key, images, labels)
            n_samples = len(shuffled_images)
            n_batches = n_samples // batch_size
            batches = (
                shuffled_images[: n_batches * batch_size].reshape(
                    n_batches, batch_size, 1, 28, 28
                ),
                shuffled_labels[: n_batches * batch_size].reshape(
                    n_batches, batch_size
                ),
            )

            initial_state = (params, opt_state)
            (params, opt_state), losses = scan(update_fn_scan, initial_state, batches)
            save_params(f"params_{epoch}.npz", params)
            print(f"Params {epoch} saved")

        # without scan it takes 11s to process one epoch
        # with scan it takes 2 minutes
        # for i in (pbar := tqdm(range(0, len(shuffled_images), batch_size))):
        #     batch_images = shuffled_images[i : i + batch_size]
        #     batch_labels = shuffled_labels[i : i + batch_size]
        #     params, loss, opt_state = update_fn(
        #         params, (batch_images, batch_labels), opt_state
        #     )
        #     losses.append(loss)
        #
        #     pbar.set_description(f"Loss: {loss:.3f}")
        # save_params(f"params_{epoch}.npz", params)
        # print(f"Params {epoch} saved")

        return params, losses

    return init_model, train_model, predict
