# AIGS project 3 - Simple classifier

The goal of this project was to implement a simple MNIST classifier using JAX, and some advanced features such as scan, vmap, and jit.
Also, I save the activations of one kernel to generate a GIF image of the activations over the epochs.
There is a naive version, and a slightly more advanced one, where I use vmap and scan.

## How to run

1. Install the requirements

```bash
pip install -r requirements.txt
```

Then we can train the model, test the accuracy, or generate GIFs.
The training is done by running `python3 main.py train` (you don't say).
To test the accuracy we can run `python3 main.py test` and finally, to generate some gifs (I picked different channels across 3 different kernel layers) we can run `python3 main.py gif`.

```

```
