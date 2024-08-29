# Scaling Up Joint Training with Encoder Sampling

## 1. The idea

Have a bunch of encoders, as well as a dataset embedded by them. Now sample some `m` encoders at each step
who will provide their image embeddings for that particular step of joint hyper-network training.

```python
for epoch in range(num_epochs):
  for step, images in enumerate(image_dataset_loader()):
    encoders = next(encoder_loader)  # minibatch of encoders
    embeddings = encoders(images)
    loss = ...
    loss.backward()
    opt.step()
```

Notes:

1. Since the embeddings are pre-computed, the entire embedding set across all image encoders will be
of shape: `[num_total_image_encoders, dataset_length, embedding_dim]`.

2. To avoid loading a `B` sized subset of this onto the RAM as an extra entity when the `data_loader` is called,
we can instead take a loader of indices, get the indices of the data samples for the curent batch.

3. Parallely, we can have `encoder_loader` which tells us which encoders to choose at the indices mentioned above.
