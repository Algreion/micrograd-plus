
# micrograd-plus

Tiny automatic differentiation engine and neural network library, built from scratch in Python (using only the Python standard library), based on Andrej Karpathy's micrograd lecture.

The goal was to implement backpropagation by hand to verify a network built from basic operations can learn. The same principles, applied to tensors, are what libraries like PyTorch automate.

## Description
- Autograd engine (Value): Every number is a node in a computation graph. Every operation is recorded, so backward() can walk the graph in reverse to compute the gradients.
- Neural networks (Neuron, Layer, MLP): Multilayer perceptron built out of Value objects, so training requires no external libraries.
- Helper functions (mse, bce, cce, one_hot, softmax, train_step): Simplified model training.
- Save and load: Model weights can be saved to a text file and loaded back, with a check that the layers match.

## Requirements
No dependencies needed.

## Usage
```python
import random
from micrograd_plus import Value, MLP, train_step

# Autograd engine
a, b = Value (-2.0), Value(3.0)
c = a*b + a.tanh()
c.backward()
print(a.grad, b.grad) # 3.07, -2.0 (dc/da, dc/db)

# Train a network on XOR
random.seed(42)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]
nn = MLP(2, [4, 4, 1], last_activation="sigmoid")
for _ in range(1000):
    train_step(nn, X, Y, cost='bce')
print([round(nn(x)) for x in X]) # [0,1,1,0]

nn.save("model.txt") # Reload weights with nn.load("model.txt")

```

## Toy problem: training on XOR
The function demo_xor(...) trains a full MLP to compute the XOR function.

Single-output learns XOR, while multi-output learns three pairwise XORs. As XOR is a non-linear function, it is a good toy problem to test the library.

The results over 30 runs with demo_test(seed=42) and default demo_xor settings are:
- Single-output: 100.00% average accuracy, 30/30 perfect runs.
- Multi-output: 93.61% average accuracy, 12/30 perfect runs.

Single-output learns XOR reliably, while multi-output reaches high accuracy but doesn't converge to a perfect solution within 1000 iterations.

## Known limitations
- backward() is recursive, graph depth is limited by Python's recursion limit (around 1000).
- The logarithm implementation fails with Value(0), so cross-entropy loss may fail if predicted probabilities are saturated.
- As the project uses pure Python scalars, it is much slower than specialized tensor libraries.
- The project focuses on the fundamentals of deep learning, so modern techniques like AdamW or GELU are not implemented.

## Credits
The project was based on Andrej Karpathy's [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) lecture, and extends the idea with additional operations and activations, model saving/loading, helper functions, and a XOR demo that trains a network end to end.
