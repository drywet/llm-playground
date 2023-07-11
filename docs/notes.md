### PyTorch notes

```
x = torch.from_numpy(np.array([
    [
        [4,5,6],
        [7,8,9],
    ],
    [
        [10,9,12],
        [13,14,15],
    ],
])).float()

x = torch.from_numpy(np.array([
    [4,5,6],
    [7,8,9],
])).float()

torch.nn.Softmax(dim=n)(x)
# dim=-1 is along rows
# dim=-2 is along columns
# dim=-3 is along batches
```

### HF transformers

```
tokenizer(...).input_ids
model.generate(...)[0]
model(...).logits
```

### Keras

Keras Embedding layer takes just one integer (token index) as input
![embedding-keras.png](embedding-keras.png)

[StackOverflow](https://stackoverflow.com/questions/47868265/what-is-the-difference-between-an-embedding-layer-and-a-dense-layer)
![embedding-vs-dense.png](embedding-vs-dense.png)

When we use embedding layer, it is generally to reduce one-hot input vectors (sparse) to denser representations.

1. Embedding layer is much like a table lookup. When the table is small, it is fast.
2. When the table is large, table lookup is much slower. In practice, we would use dense layer as a dimension reducer to
   reduce the one-hot input instead of embedding layer in this case.

#### Metrics

categorical_crossentropy input is one-hot vector
sparse_categorical_crossentropy input is just an index of a category

#### SGD, batch_size

A smaller batch_size + more training steps can be more accurate than a bigger batch_size + fewer updates. 

There are several ways to understand why several updates is better (for the same amount of data being read). It's the
key idea of stochastic gradient descent vs. gradient descent. Instead of reading everything and then correct yourself at
the end, you correct yourself on the way, making the next reads more useful since you correct yourself from a better
guess. Geometrically, several updates is better because you are drawing several segments, each in the direction of the (
approximated) gradient at the start of each segment. while a single big update is a single segment from the very start
in the direction of the (exact) gradient. It's better to change direction several times even if the direction is less
precise.


