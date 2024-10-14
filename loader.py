import jax.numpy as jnp


def save_params(filename, params):
    with open(filename, "wb") as wf:
        conv = params["conv"]
        dense = params["dense"]

        jnp.savez(
            wf,
            w1=dense[0][0],
            b1=dense[0][1],
            w2=dense[1][0],
            b2=dense[1][1],
            w3=dense[2][0],
            b3=dense[2][1],
            kernel0=conv[0],
            kernel1=conv[1],
            kernel2=conv[2],
        )


def load_params(filename):
    weights = []
    kernels = []

    rf = open(filename, "rb")
    saved = jnp.load(rf)

    weights.append((saved["w1"], saved["b1"]))
    weights.append((saved["w2"], saved["b2"]))
    weights.append((saved["w3"], saved["b3"]))

    kernels.append(saved["kernel0"])
    kernels.append(saved["kernel1"])
    kernels.append(saved["kernel2"])

    rf.close()
    return {"conv": kernels, "dense": weights}
