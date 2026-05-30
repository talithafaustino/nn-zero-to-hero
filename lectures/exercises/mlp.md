## Questions

These exercises follow the chronological flow of the **makemore Part 2: MLP** video and the notebook [makemore/makemore_part2_mlp.ipynb](makemore/makemore_part2_mlp.ipynb). Work through them in order if you want to re-create the notebook from scratch.

---

### Part 1: Data and Context Dataset

**Exercise 1: Imports and Dataset Loading**  
Write code to:
- Import `torch`, `torch.nn.functional as F`, and `matplotlib.pyplot as plt`.
- Enable inline plotting for notebooks.
- Read the `names.txt` file and put all names in a Python list.
- Inspect the first few names and the total number of names.

**Exercise 2: Character Vocabulary and Mappings**  
Build a character-level vocabulary:
- Collect all unique characters in the dataset.
- Sort them and construct two dictionaries:
  - `stoi`: maps each character to an integer index, reserving `0` for the special `'.'` token.
  - `itos`: inverse mapping from index to character.
- Print `itos` to verify the mapping.

**Exercise 3: Building the (Context, Next-Char) Dataset**  
Create a dataset of fixed-length contexts and next characters:
- Set `block_size = 3` (three characters of context).
- For every word `w` in `words`, iterate through `w + '.'` while maintaining a rolling context of length 3, initialized as `[0, 0, 0]`.
- For each character, append the current context to a list `X` and the current character index to a list `Y`, then update the context by cropping and appending the new index.
- Convert `X` and `Y` to `torch.tensor` objects and inspect their shapes and dtypes.

---

### Part 2: Tiny Embedding and One Hidden Layer

**Exercise 4: Character Embedding Table (2D)**  
Introduce a small embedding table:
- Create a tensor `C` of shape `(27, 2)` initialized with random values.
- Use integer indexing with `X` to obtain `emb = C[X]`.
- Inspect the shape of `emb` and confirm it is `(num_examples, 3, 2)`.

**Exercise 5: Hidden Layer and tanh Nonlinearity**  
Build a single hidden layer on top of the embeddings:
- Create weights `W1` of shape `(6, 100)` and biases `b1` of shape `(100,)` using random initialization.
- Flatten the last two dimensions of `emb` into shape `(-1, 6)` using `.view` and compute hidden activations `h = torch.tanh(emb.view(-1, 6) @ W1 + b1)`.
- Inspect `h` and its shape to verify it is `(num_examples, 100)`.

**Exercise 6: Output Layer, Softmax, and Manual NLL Loss**  
Complete the forward pass for a shallow MLP on a small batch:
- Define `W2` with shape `(100, 27)` and `b2` with shape `(27,)`.
- Compute `logits = h @ W2 + b2` and inspect its shape.
- Turn `logits` into probabilities:
  - `counts = logits.exp()`
  - `prob = counts / counts.sum(1, keepdims=True)`
- For the first 32 examples, compute the negative log-likelihood loss:
  - Index the probabilities of the correct next character using `prob[torch.arange(32), Y[:32]]`.
  - Take `log`, negate, and average to obtain a scalar `loss`.

---

### Part 3: Train / Dev / Test Split and Dataset Helper

**Exercise 7: Dataset Builder Function**  
Refactor dataset creation into a reusable function:
- Keep `block_size` and the `stoi/itos` mappings from earlier.
- Implement a function `build_dataset(words)` that:
  - Builds lists `X` and `Y` using the same context-rolling logic as in Exercise 3.
  - Converts them to tensors, prints their shapes, and returns `(X, Y)`.

**Exercise 8: Shuffling and Splitting Words**  
Create the standard 80/10/10 train-dev-test split:
- Set a Python `random` seed for reproducibility.
- Shuffle the list `words` in-place.
- Compute indices `n1 = int(0.8 * len(words))` and `n2 = int(0.9 * len(words))`.
- Build three datasets using `build_dataset`:
  - `Xtr, Ytr = build_dataset(words[:n1])`
  - `Xdev, Ydev = build_dataset(words[n1:n2])`
  - `Xte, Yte = build_dataset(words[n2:])`

---

### Part 4: Respectable MLP: Parameters and Training Loop

**Exercise 9: Model Parameters and Counting Them**  
Move to the final network architecture used in the video:
- Create a `torch.Generator` with a fixed manual seed.
- Initialize the parameters:
  - `C` of shape `(27, 10)`
  - `W1` of shape `(30, 200)`
  - `b1` of shape `(200,)`
  - `W2` of shape `(200, 27)`
  - `b2` of shape `(27,)`
- Put all of them into a list `parameters`.
- Compute and print the total number of parameters using `sum(p.nelement() for p in parameters)`.

**Exercise 10: Enabling Gradients and Learning-Rate Range**  
Prepare for gradient-based training:
- Loop over all `parameters` and set `p.requires_grad = True`.
- Create a logarithmically spaced learning-rate range for exploration:
  - `lre = torch.linspace(-3, 0, 1000)`
  - `lrs = 10**lre`
- Create empty lists to track training statistics: `lri`, `lossi`, `stepi`.

**Exercise 11: Mini-batch Training Loop with LR Schedule**  
Implement the main training loop:
- For 200,000 iterations:
  - Sample a mini-batch of indices `ix` of size 32 from the range `[0, Xtr.shape[0])`.
  - Forward pass:
    - `emb = C[Xtr[ix]]` (shape `(32, 3, 10)`).
    - `h = torch.tanh(emb.view(-1, 30) @ W1 + b1)` (shape `(32, 200)`).
    - `logits = h @ W2 + b2` (shape `(32, 27)`).
    - `loss = F.cross_entropy(logits, Ytr[ix])`.
  - Backward pass:
    - Zero all gradients by setting `p.grad = None` for each `p` in `parameters`.
    - Call `loss.backward()`.
  - Parameter update with a piecewise-constant learning rate:
    - Use `lr = 0.1` for the first 100,000 steps and `lr = 0.01` afterwards.
    - Update each parameter with `p.data += -lr * p.grad`.
  - Append the iteration index and `loss.log10().item()` to `stepi` and `lossi` respectively.

---

### Part 5: Evaluation, Visualization, and Sampling

**Exercise 12: Plotting Training Loss**  
Visualize how the loss evolves:
- Use `plt.plot(stepi, lossi)` to plot the log10-loss over training steps.

**Exercise 13: Train and Dev Loss Evaluation**  
Evaluate the final trained model:
- Compute the training loss using the entire training set `Xtr, Ytr` with the current parameters and `F.cross_entropy`.
- Compute the dev/validation loss on `Xdev, Ydev` in the same way.
- Print both losses to compare them.

**Exercise 14: Visualizing the Character Embeddings**  
Visualize the first two dimensions of the embedding matrix:
- Create a scatter plot of `C[:, 0]` vs `C[:, 1]`.
- Overlay each point with its corresponding character label (using `itos[i]`).
- Add a grid for easier reading.

**Exercise 15: Sampling Names from the Model**  
Implement sampling from the trained model, mirroring the video:
- Set up a new `torch.Generator` with a fixed seed (e.g., `2147483647 + 10`).
- For 20 samples:
  - Start with `context = [0] * block_size`.
  - In a loop:
    - Embed the current context with `C[torch.tensor([context])]`.
    - Run it through the MLP to obtain `logits`.
    - Convert `logits` to probabilities with `F.softmax(logits, dim=1)`.
    - Sample the next index with `torch.multinomial`.
    - Update the context by dropping the oldest index and appending the new one.
    - Append the new index to an output list and stop when the sampled index is 0.
  - Decode and print the name by translating indices to characters with `itos`.

---

### Part 6: Hyperparameter Exploration Challenge

**Exercise 16 (Open-Ended): Beat the Dev Loss**  
Using the final code as a baseline, try to improve the dev loss (around ~2.17 in the video). Possible knobs include:
- Hidden layer width (size of `W1` and `b1`).
- Embedding dimensionality (size of `C`).
- Context length `block_size`.
- Batch size, training duration, and learning-rate schedule.
- Regularization or other architectural tweaks suggested in the Bengio et al. 2003 paper.


## Answers

Below are reference implementations for the main exercises, closely following the notebook [makemore/makemore_part2_mlp.ipynb](makemore/makemore_part2_mlp.ipynb). Minor differences in printing, comments, or temporary variables are fine as long as the logic and shapes match.

---

### Part 1: Data and Context Dataset — Answers

**Exercise 1 Answer: Imports and Dataset Loading**
```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt  # for making figures
%matplotlib inline

# read in all the words
words = open('names.txt', 'r').read().splitlines()
words[:8]

len(words)
```

**Exercise 2 Answer: Character Vocabulary and Mappings**
```python
# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
print(itos)
```

**Exercise 3 Answer: Building the (Context, Next-Char) Dataset**
```python
# build the dataset
block_size = 3  # context length: how many characters do we take to predict the next one?
X, Y = [], []
for w in words:

  # print(w)
  context = [0] * block_size
  for ch in w + '.':
    ix = stoi[ch]
    X.append(context)
    Y.append(ix)
    # print(''.join(itos[i] for i in context), '--->', itos[ix])
    context = context[1:] + [ix]  # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)

X.shape, X.dtype, Y.shape, Y.dtype
```

---

### Part 2: Tiny Embedding and One Hidden Layer — Answers

**Exercise 4 Answer: Character Embedding Table (2D)**
```python
C = torch.randn((27, 2))

emb = C[X]
emb.shape
```

**Exercise 5 Answer: Hidden Layer and tanh Nonlinearity**
```python
W1 = torch.randn((6, 100))
b1 = torch.randn(100)

h = torch.tanh(emb.view(-1, 6) @ W1 + b1)

h
h.shape
```

**Exercise 6 Answer: Output Layer, Softmax, and Manual NLL Loss**
```python
W2 = torch.randn((100, 27))
b2 = torch.randn(27)

logits = h @ W2 + b2
logits.shape

counts = logits.exp()
prob = counts / counts.sum(1, keepdims=True)
prob.shape

# manual negative log-likelihood on a small batch (first 32 examples)
loss = -prob[torch.arange(32), Y[:32]].log().mean()
loss
```

---

### Part 3: Train / Dev / Test Split and Dataset Helper — Answers

**Exercise 7 Answer: Dataset Builder Function**
```python
# build the dataset
block_size = 3  # context length: how many characters do we take to predict the next one?

def build_dataset(words):
  X, Y = [], []
  for w in words:

    # print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      # print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix]  # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y
```

**Exercise 8 Answer: Shuffling and Splitting Words**
```python
import random
random.seed(42)
random.shuffle(words)

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

Xtr.shape, Ytr.shape  # dataset
```

---

### Part 4: Respectable MLP: Parameters and Training Loop — Answers

**Exercise 9 Answer: Model Parameters and Counting Them**
```python
g = torch.Generator().manual_seed(2147483647)  # for reproducibility

C = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]

sum(p.nelement() for p in parameters)  # number of parameters in total
```

**Exercise 10 Answer: Enabling Gradients and Learning-Rate Range**
```python
for p in parameters:
  p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri = []
lossi = []
stepi = []
```

**Exercise 11 Answer: Mini-batch Training Loop with LR Schedule**
```python
for i in range(200000):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (32,))

  # forward pass
  emb = C[Xtr[ix]]              # (32, 3, 10)
  h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 200)
  logits = h @ W2 + b2          # (32, 27)
  loss = F.cross_entropy(logits, Ytr[ix])
  # print(loss.item())

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  # lr = lrs[i]
  lr = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  # lri.append(lre[i])
  stepi.append(i)
  lossi.append(loss.log10().item())

# print(loss.item())
```

---

### Part 5: Evaluation, Visualization, and Sampling — Answers

**Exercise 12 Answer: Plotting Training Loss**
```python
plt.plot(stepi, lossi)
```

**Exercise 13 Answer: Train and Dev Loss Evaluation**
```python
# training loss
emb = C[Xtr]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytr)
loss

# dev/validation loss
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)
loss
```

**Exercise 14 Answer: Visualizing the Character Embeddings**
```python
plt.figure(figsize=(8, 8))
plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i, 0].item(), C[i, 1].item(), itos[i],
             ha="center", va="center", color='white')
plt.grid('minor')
```

**Exercise 15 Answer: Sampling Names from the Model**
```python
# training split, dev/validation split, test split
# 80%, 10%, 10%

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):

  out = []
  context = [0] * block_size  # initialize with all dots
  while True:
    emb = C[torch.tensor([context])]      # (1, block_size, d)
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    context = context[1:] + [ix]
    out.append(ix)
    if ix == 0:
      break

  print(''.join(itos[i] for i in out))
```

---

### Part 6: Hyperparameter Exploration Challenge — Answer Sketch

**Exercise 16 Answer (High-Level Guidance)**

There is no single correct code answer here, but typical successful modifications include:
- Increasing the hidden size (e.g., 300–500 units) while monitoring overfitting via the dev loss.
- Increasing the embedding dimension beyond 10 (e.g., 20, 32, or 64) to give the model more representational capacity.
- Increasing `block_size` to use longer contexts (e.g., 4, 5, or more characters).
- Adjusting batch size (e.g., 64 or 128) and the number of training iterations.
- Trying different learning-rate schedules (e.g., cosine decay or more gradual step decay).

The key is to:
- Train only on `Xtr, Ytr`.
- Use `Xdev, Ydev` to choose hyperparameters.
- Evaluate `Xte, Yte` only once you are satisfied with your dev performance.