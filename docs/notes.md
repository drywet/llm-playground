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

https://stackoverflow.com/questions/47868265/what-is-the-difference-between-an-embedding-layer-and-a-dense-layer
When we use embedding layer, it is generally to reduce one-hot input vectors (sparse) to denser representations.

1. Embedding layer is much like a table lookup. When the table is small, it is fast.
2. When the table is large, table lookup is much slower. In practice, we would use dense layer as a dimension reducer to
   reduce the one-hot input instead of embedding layer in this case.

![embedding-vs-dense.png](embedding-vs-dense.png)