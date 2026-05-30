## Questions

These exercises follow the chronological flow of the **Let's build GPT: from scratch, in code, spelled out** video and the reference scripts [nanogpt/bigram.py](../nanogpt/bigram.py) and [nanogpt/gpt.py](../nanogpt/gpt.py). Work through them in order to build a decoder-only Transformer language model from an empty file, starting with a bigram baseline and ending with a scaled-up GPT generating Shakespeare-like text.

---

### Part 1: Data Loading and Tokenization

**Exercise 1: Read and Inspect the Dataset**  
Write the setup code:
- Import `torch`, `torch.nn as nn`, and `torch.nn.functional as F`.
- Read the tiny Shakespeare text file into a string variable.
- Print the total number of characters in the dataset and the first 1,000 characters.
- Create a sorted list of all unique characters in the text and print the vocabulary size. You should find 65 distinct characters.

**Exercise 2: Character-Level Tokenizer**  
Build an encoder and decoder:
- Create `stoi` (character → integer) and `itos` (integer → character) mappings.
- Write an `encode` function that takes a string and returns a list of integers.
- Write a `decode` function that takes a list of integers and returns a string.
- Verify round-trip correctness: `decode(encode("hi there"))` should return `"hi there"`.

**Exercise 3: Train / Validation Split**  
Tokenize the entire dataset and split it:
- Encode the full text into a `torch.tensor` of dtype `torch.long`. Call it `data`.
- Print the shape and the first 1,000 elements.
- Split: first 90% is `train_data`, last 10% is `val_data`.

**Exercise 4: Batching — `get_batch`**  
Implement a `get_batch(split)` function:
- Set `block_size = 8` (maximum context length) and `batch_size = 4`.
- Seed with `torch.manual_seed(1337)`.
- For a given split (`'train'` or `'val'`), generate `batch_size` random starting indices into the data.
- Stack `block_size`-length chunks into `x` (inputs) and `y` (targets, offset by 1).
- Print the shapes of `xb` and `yb` (`(4, 8)` each).
- Spell out the 32 individual examples packed in this batch: for each row, for each time step `t`, print the context `x[b, :t+1]` and the target `y[b, t]`.

Explain why we train on all context lengths from 1 up to `block_size`, not just the full context.

---

### Part 2: Bigram Language Model Baseline

**Exercise 5: Implementing `BigramLanguageModel`**  
Create an `nn.Module` subclass called `BigramLanguageModel`:
- In `__init__`, create `self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)`.
- In `forward(self, idx, targets=None)`:
  - Look up embeddings: `logits = self.token_embedding_table(idx)` → shape `(B, T, C)`.
  - If `targets` is provided, reshape logits to `(B*T, C)` and targets to `(B*T)`, then compute `F.cross_entropy(logits, targets)`.
  - Return `(logits, loss)` where `loss` is `None` when targets are not given.
- Print the logits shape and the loss for a single batch.

Verify that the initial loss is around 4.87. Explain why the expected loss for a uniform 65-class distribution is $-\ln(1/65) \approx 4.17$ and why we're higher than that.

**Exercise 6: Generation from the Bigram Model**  
Add a `generate(self, idx, max_new_tokens)` method:
- Starting from `idx` (shape `(B, T)`), loop `max_new_tokens` times:
  - Forward the current sequence to get logits.
  - Focus on the last time step: `logits[:, -1, :]` → `(B, C)`.
  - Apply `F.softmax` to get probabilities.
  - Sample from the distribution with `torch.multinomial(probs, num_samples=1)`.
  - Concatenate the new token to the running sequence.
- Return the extended sequence.

Generate 100 tokens starting from a single `0` token (the newline character). The output should be gibberish since the model is untrained.

**Exercise 7: Training the Bigram Model**  
Train the bigram model:
- Create an `AdamW` optimizer with `lr=1e-3`.
- Run a training loop for 10,000 steps with `batch_size = 32`:
  - Sample a batch, compute the loss, zero gradients, backpropagate, step.
- The loss should converge to roughly 2.5.
- Generate 500 tokens and observe improved (though still poor) output.

---

### Part 3: Packaging into a Script and Loss Estimation

**Exercise 8: `estimate_loss` and Device Handling**  
Refactor the training code into a clean script form:
- Add GPU support: `device = 'cuda' if torch.cuda.is_available() else 'cpu'`. Move data and model to `device`.
- Implement `estimate_loss()`:
  - Decorated with `@torch.no_grad()`.
  - Set the model to `eval()` mode.
  - For each split (`'train'`, `'val'`), average the loss over `eval_iters = 200` batches.
  - Set the model back to `train()` mode.
  - Return a dict of average losses.
- In the training loop, call `estimate_loss()` every `eval_interval = 300` steps and print the results.
- Train for `max_iters = 3000` with `batch_size = 32`, `block_size = 8`, `lr = 1e-2`.

Explain why we switch to `model.eval()` and use `torch.no_grad()` during loss estimation, even though our current model has no layers that behave differently between train and eval modes.

---

### Part 4: The Mathematical Trick — Weighted Aggregation via Matrix Multiply

**Exercise 9: Averaging Past Tokens (Naive Loop)**  
Before implementing self-attention, build intuition for how tokens can communicate:
- Create a random tensor `x` of shape `(B, T, C)` with `B=4, T=8, C=2`.
- Using a nested for loop over `b` and `t`, compute `xbow[b, t]` as the mean of `x[b, :t+1]` — i.e., the average of all tokens from position 0 up to and including position `t`.
- Verify that `xbow[0]` shows the progressive averaging: position 0 equals `x[0, 0]`, position 1 is the mean of `x[0, :2]`, etc.

**Exercise 10: Vectorizing with Matrix Multiplication**  
Replace the loop with a batched matrix multiply:
- Create a lower triangular matrix using `torch.tril(torch.ones(T, T))`.
- Normalize each row so it sums to 1 (divide by the row sum).
- Multiply this weight matrix by `x` using `@` — PyTorch broadcasts across the batch dimension.
- Verify `torch.allclose(xbow, xbow2)`.

Explain why the lower triangular structure enforces causality (no future information leaks into the past).

**Exercise 11: Using Softmax for the Weights**  
Rewrite the aggregation a third way, using softmax — this previews the self-attention pattern:
- Start with `wei = torch.zeros((T, T))`.
- Use `wei.masked_fill(tril == 0, float('-inf'))` to set future positions to negative infinity.
- Apply `F.softmax(wei, dim=-1)` to normalize.
- Multiply `wei @ x` and verify the result is identical to the previous two versions.

Explain why this third form is the one used in self-attention: the initial zeros will be replaced by data-dependent affinity scores (queries dot-producted with keys), and the masking + softmax will turn them into a proper probability distribution over the past.

---

### Part 5: Self-Attention

**Exercise 12: Positional and Token Embeddings**  
Before implementing attention, set up the embedding infrastructure:
- Introduce `n_embd = 32`. Change the token embedding table to map `vocab_size → n_embd` instead of `vocab_size → vocab_size`.
- Add a position embedding table: `nn.Embedding(block_size, n_embd)`.
- In `forward`, create positional embeddings using `torch.arange(T, device=device)` and add them to the token embeddings.
- Add a language modeling head: `nn.Linear(n_embd, vocab_size)` to project from `n_embd` back to logits.
- In `generate`, crop the input to the last `block_size` tokens before forwarding (to respect the position embedding table's range).

Explain why positional embeddings are necessary: attention is permutation-equivariant and has no built-in notion of position, so without them the model cannot distinguish token order.

**Exercise 13: Implementing a Single Head of Self-Attention**  
Create a `Head` module:
- `__init__` takes `head_size`. Creates three `nn.Linear(n_embd, head_size, bias=False)` layers for key, query, and value. Registers a buffer `tril` — the lower triangular mask.
- `forward(self, x)`:
  - Compute `k = self.key(x)` and `q = self.query(x)` — both `(B, T, head_size)`.
  - Compute attention scores: `wei = q @ k.transpose(-2, -1)` → `(B, T, T)`.
  - **Scale** by `* head_size**-0.5` (the $1/\sqrt{d_k}$ factor).
  - Mask future positions with `masked_fill`.
  - Apply `F.softmax(wei, dim=-1)`.
  - Compute output: `wei @ self.value(x)` → `(B, T, head_size)`.
- Plug a single `Head(n_embd)` into the language model and train.

Explain in detail:
1. What query, key, and value represent intuitively: query = "what am I looking for?", key = "what do I contain?", value = "what will I communicate if you find me interesting?".
2. Why the $1/\sqrt{d_k}$ scaling is critical: without it, the dot products grow with head size, pushing softmax towards one-hot vectors at initialization. This means each token would aggregate from essentially one other token, destroying the gradient signal. Scaling keeps the variance of the attention scores at ~1, ensuring a diffuse softmax.
3. Why this is called "self"-attention: queries, keys, and values all come from the same source `x`. In cross-attention, keys and values come from a different source (e.g., an encoder).

**Exercise 14: Multi-Head Attention**  
Create a `MultiHeadAttention` module:
- Takes `num_heads` and `head_size`.
- Creates `num_heads` independent `Head(head_size)` modules using `nn.ModuleList`.
- In `forward`, runs all heads in parallel, concatenates their outputs along the channel dimension.
- Add a projection layer: `nn.Linear(head_size * num_heads, n_embd)` — this projects back into the residual pathway.
- Replace the single head with `MultiHeadAttention(4, n_embd // 4)` — 4 heads of 8-dimensional attention.

Explain why multiple heads help: each head can specialize in a different type of relationship (e.g., one head looks for vowels, another for positional patterns). The concatenation combines all these different perspectives.

---

### Part 6: Feed-Forward Network

**Exercise 15: Adding a Feed-Forward Layer**  
Create a `FeedForward` module:
- A small MLP: `nn.Linear(n_embd, 4 * n_embd)` → `nn.ReLU()` → `nn.Linear(4 * n_embd, n_embd)`.
- The inner layer is 4× wider than the embedding dimension (following the original paper: 512 → 2048 → 512).
- Apply this after self-attention in the forward pass: tokens first communicate (attention), then individually compute (feed-forward).

Explain the role of the feed-forward layer: self-attention is the communication step where tokens exchange information, but they need time to "think" about what they've gathered. The feed-forward network applies per-token computation — it processes each position independently and identically.

---

### Part 7: Transformer Block, Residual Connections, and Layer Norm

**Exercise 16: The Transformer Block**  
Create a `Block` module that combines communication and computation:
- `__init__` takes `n_embd` and `n_head`. Computes `head_size = n_embd // n_head`.
- Contains a `MultiHeadAttention` and a `FeedForward`.
- In `forward`: apply attention, then feed-forward.
- Stack multiple blocks using `nn.Sequential`.

**Exercise 17: Residual (Skip) Connections**  
Add residual connections to the block:
- Change the forward pass to: `x = x + self.sa(x)` and `x = x + self.ffwd(x)`.
- The `+` means we fork off from the residual pathway, perform computation, and add the result back.

Explain why residual connections are essential for deep networks: during backpropagation, the addition distributes gradients equally to both branches. This creates a "gradient superhighway" that flows directly from the loss to the input, unimpeded by the depth of the network. The residual blocks are initialized to contribute very little at first, then gradually "come online" during training. Without skip connections, deep transformers are very hard to optimize.

**Exercise 18: Layer Normalization**  
Add layer normalization using the **pre-norm** formulation (which departs slightly from the original paper):
- Create two `nn.LayerNorm(n_embd)` instances in the block.
- Apply them *before* the attention and feed-forward, respectively:
  ```
  x = x + self.sa(self.ln1(x))
  x = x + self.ffwd(self.ln2(x))
  ```
- Add a final `nn.LayerNorm(n_embd)` in the model, applied after all blocks and before the language modeling head.

Explain the difference between BatchNorm and LayerNorm:
- BatchNorm normalizes across the batch dimension (each neuron's statistics are computed over all examples in the batch). It has different behavior at train vs. test time and requires running buffers.
- LayerNorm normalizes across the feature dimension (each example's statistics are computed over all features). There is no distinction between train and test time, no running buffers, and no coupling across batch elements.
- In the Transformer, LayerNorm normalizes each token's feature vector independently — the mean and variance are computed over the `n_embd` dimensions.

---

### Part 8: Dropout and Scaling Up

**Exercise 19: Adding Dropout**  
Add `nn.Dropout(dropout)` at three locations:
1. After the softmax in the attention head (drops out some attention connections).
2. After the projection in multi-head attention (before the residual connection).
3. After the ReLU in the feed-forward network (before the residual connection).

Set `dropout = 0.2`. Explain dropout as a regularization technique: each forward pass randomly zeroes out a subset of activations, effectively training an ensemble of sub-networks. At test time, all activations are used, merging the ensemble.

**Exercise 20: Scaling Up and Final Training**  
Scale up the model to the final configuration:
- `batch_size = 64`, `block_size = 256`, `n_embd = 384`, `n_head = 6`, `n_layer = 6`, `dropout = 0.2`.
- `lr = 3e-4`, `max_iters = 5000`, `eval_interval = 500`.
- Print the total number of parameters (should be ~10.7M).
- Train the model (requires a GPU; ~15 minutes on an A100).
- The validation loss should reach roughly **1.48**.
- Generate 500+ tokens and observe Shakespeare-like output: characters speak in dialogue format, lines are structured, vocabulary is plausible, though content is nonsensical.

---

### Part 9: Encoder vs. Decoder and the Road to ChatGPT

**Exercise 21: Understanding the Full Transformer Architecture**  
This is a conceptual exercise — no code changes needed. Answer the following:

1. **Why is our model a "decoder-only" Transformer?** We use the triangular causal mask to prevent future tokens from attending to past tokens. This autoregressive structure lets us generate text left-to-right. We have no encoder and no cross-attention.

2. **What would an encoder block look like?** Remove the causal mask — all tokens attend to all other tokens. This is useful when the entire input is available (e.g., sentiment analysis, classification).

3. **What is cross-attention?** In an encoder-decoder Transformer (like the original paper's machine translation model), the decoder's queries come from the decoder's own tokens, but the keys and values come from the encoder's output. This conditions the generation on some external context.

4. **How does ChatGPT relate to what we built?** ChatGPT uses the same decoder-only architecture, but with ~1000× more parameters, trained on ~1,000,000× more data (hundreds of billions of tokens from the internet). After pre-training, it undergoes fine-tuning stages (supervised fine-tuning on instruction-response pairs, then RLHF with a reward model) to become an aligned assistant rather than just a document completer.

---

### Part 10: Going Further (Karpathy's Challenges)

**Exercise 22: Batched Multi-Head Attention**  
The n-dimensional tensor mastery challenge. Right now our `MultiHeadAttention` class creates `n_head` separate `Head` modules and loops through them:
```python
out = torch.cat([h(x) for h in self.heads], dim=-1)
```
This is clean but inefficient — each head runs sequentially in Python. Combine `Head` and `MultiHeadAttention` into a single `CausalSelfAttention` class that processes **all heads in parallel** by treating the head dimension as another batch dimension:
- Use a single `nn.Linear(n_embd, 3 * n_embd)` to produce queries, keys, and values for all heads at once.
- Reshape the `(B, T, n_embd)` output into `(B, n_head, T, head_size)` so each head is an independent "batch".
- Compute the attention scores, masking, and softmax on this 4D tensor.
- Reassemble the heads back into `(B, T, n_embd)` and apply the output projection.

The answer follows the pattern used in the real nanoGPT `model.py`.

**Exercise 23: Train on Your Own Dataset**  
Replace tiny Shakespeare with a dataset of your choice. Some fun ideas:
- Song lyrics, poetry, recipes, code, LaTeX papers, band names, legal documents.
- **Advanced challenge — addition**: train a GPT to do addition (`a+b=c`). Tips:
  - Predict the digits of `c` in reverse order (matching how the standard addition algorithm proceeds right-to-left).
  - Modify the data loader to generate random problems on the fly (skip `train.bin` / `val.bin`).
  - Mask out the loss on the input positions (`a+b=`) by setting those targets to `-1` and using `CrossEntropyLoss(ignore_index=-1)`.
  - Does your Transformer learn to add? If yes, extend to all of `+-*/` for a full calculator. You may need to generate Chain-of-Thought traces to teach multi-step reasoning.

**Exercise 24: Pretraining and Fine-Tuning**  
Find a dataset that is very large — large enough that you see no gap between train and val loss (i.e., the model underfits).
- Pretrain the Transformer on this large dataset.
- Save the model checkpoint.
- Initialize from that checkpoint and fine-tune on tiny Shakespeare with a smaller number of steps and a lower learning rate.
- Compare the validation loss against training from scratch. Can you beat the scratch-trained model through pretraining + fine-tuning?

**Exercise 25: Read and Implement a Transformer Improvement**  
Pick one idea from the Transformer literature and implement it in your GPT. Some suggestions:
- **Rotary Position Embeddings (RoPE)** — replaces learned absolute position embeddings.
- **SwiGLU activation** — replaces ReLU in the feed-forward layer.
- **Flash Attention** — use `torch.nn.functional.scaled_dot_product_attention` for fused, memory-efficient attention.
- **RMSNorm** — replaces LayerNorm with a simpler, bias-free normalization.
- **Weight tying** — share the token embedding matrix with the output projection (`lm_head`).
- **Cosine learning rate schedule with warmup**.

Measure the validation loss before and after. Does your change help?

---

## Answers

---

### Part 1: Data Loading and Tokenization

**Answer 1: Read and Inspect the Dataset**

The tiny Shakespeare dataset is a ~1MB text file containing a concatenation of Shakespeare's works. It serves as our training corpus for a character-level language model.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters:", len(text))
print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
```

**Answer 2: Character-Level Tokenizer**

We use a character-level tokenizer — the simplest possible scheme. Each character maps to a unique integer. More sophisticated tokenizers (like BPE used in GPT-2 with ~50,000 tokens) trade vocabulary size for sequence length: larger vocabularies produce shorter sequences but require more embedding parameters. Character-level tokenization gives us a tiny vocabulary (65) but very long sequences.

```python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hi there"))
print(decode(encode("hi there")))
```

**Answer 3: Train / Validation Split**

We withhold 10% of the data as a validation set to detect overfitting. We want the model to learn general patterns of Shakespeare-like text, not just memorize the training data.

```python
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

**Answer 4: Batching — `get_batch`**

A single chunk of `block_size + 1` characters contains `block_size` independent training examples (contexts of length 1 through `block_size`). We train on all these sub-contexts so the Transformer learns to predict from any amount of history — not just the full context. This is critical for generation: sampling starts from a single token and the model must be able to predict from short contexts.

```python
torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")
```

---

### Part 2: Bigram Language Model Baseline

**Answer 5: Implementing `BigramLanguageModel`**

The bigram model is the simplest possible language model: it predicts the next character based solely on the identity of the current character, with no context from previous positions. The embedding table acts as a lookup of logits — each row stores the log-probability distribution over what character is likely to follow. The tokens don't "talk to each other" at all.

The initial loss of ~4.87 is higher than the theoretical $-\ln(1/65) \approx 4.17$ because the randomly initialized logits are not uniform — they have some random entropy that makes the predictions worse than chance.

```python
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

m = BigramLanguageModel()
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
```

**Answer 6: Generation from the Bigram Model**

The `generate` method extends the sequence autoregressively. At each step it takes the logits for the last position, converts them to probabilities via softmax, samples one token, and appends it. Even though we feed the entire history through the model, the bigram model only uses the last token — the full history is fed in to keep the API general for when we implement attention later.

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, loss = self(idx)
        logits = logits[:, -1, :]  # (B, C)
        probs = F.softmax(logits, dim=-1)  # (B, C)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    return idx

# Add to BigramLanguageModel, then:
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
```

**Answer 7: Training the Bigram Model**

We use AdamW instead of plain SGD. Adam maintains per-parameter running estimates of gradient mean and variance, which allows it to adaptively set learning rates. For small networks like this, a relatively high learning rate (1e-3) works well. After 10,000 steps the loss should be around 2.5 — far from perfect, but the output starts to show character-level patterns.

```python
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
```

---

### Part 3: Packaging into a Script and Loss Estimation

**Answer 8: `estimate_loss` and Device Handling**

Printing `loss.item()` inside the training loop gives a very noisy measurement because each batch is a random sample. `estimate_loss` averages the loss over many batches for a stable estimate. We use `model.eval()` as good practice — some layers (dropout, batchnorm) behave differently during evaluation. `torch.no_grad()` tells PyTorch not to build a computation graph, saving memory since we won't call `.backward()`.

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 300
max_iters = 3000
learning_rate = 1e-2

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

---

### Part 4: The Mathematical Trick — Weighted Aggregation via Matrix Multiply

**Answer 9: Averaging Past Tokens (Naive Loop)**

This is the simplest form of token communication: each token averages all previous tokens (including itself). It's a "bag of words" approach — extremely lossy because it discards all information about ordering and relative position. But it's the starting point for understanding self-attention, which replaces the uniform average with a data-dependent weighted average.

```python
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]  # (t+1, C)
        xbow[b, t] = torch.mean(xprev, 0)
```

**Answer 10: Vectorizing with Matrix Multiplication**

The key insight: matrix multiplication can perform weighted aggregation efficiently. A lower triangular weight matrix, with each row summing to 1, computes the incremental average of all preceding tokens. The lower triangular structure enforces causality — position `t` can only see positions `0` through `t`. Positions in the upper triangle are zeroed out, meaning future information cannot flow backwards.

```python
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x  # (T, T) @ (B, T, C) --> (B, T, C)
print(torch.allclose(xbow, xbow2))
```

**Answer 11: Using Softmax for the Weights**

This third formulation is the template for self-attention. We start with scores (`wei`), mask out future positions with `-inf`, and normalize with softmax. In self-attention, the initial zeros will be replaced by `q @ k.T` — data-dependent affinities between tokens. The masking ensures causality, and softmax produces a proper probability distribution. This is why the matrix-multiply trick is so important: it's the backbone of how attention aggregates information efficiently.

```python
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
print(torch.allclose(xbow, xbow3))
```

---

### Part 5: Self-Attention

**Answer 12: Positional and Token Embeddings**

Attention operates on sets — it has no built-in notion of ordering. Without positional embeddings, the model cannot distinguish "ABC" from "CBA" because the self-attention scores would be identical (just permuted). By adding a position embedding to each token embedding, we give every token a unique signature that encodes both *what* it is and *where* it is in the sequence. The `lm_head` linear layer projects from the `n_embd`-dimensional internal representation back to vocabulary-sized logits for next-token prediction.

```python
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd) — broadcasting adds batch dim
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # crop to block_size
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```

**Answer 13: Implementing a Single Head of Self-Attention**

Self-attention is the core mechanism that lets tokens communicate. Each token produces three vectors:
- **Query** ("what am I looking for?"): a learned linear projection of the token's representation.
- **Key** ("what do I contain?"): another learned linear projection.
- **Value** ("what will I communicate?"): a third projection — this is what actually gets aggregated. Think of `x` as private information; `v` is what the token is willing to share.

The dot product `q @ k.T` measures the affinity between each pair of tokens. High affinity means the query found what it was looking for in the key. The $1/\sqrt{d_k}$ scaling is critical: without it, the dot products grow proportionally to `head_size` (because each element of the sum contributes variance), pushing softmax towards one-hot outputs. At initialization, this would mean each token aggregates from essentially one other token, creating a very sparse gradient signal. Scaling keeps the pre-softmax values at unit variance, ensuring a diffuse attention distribution that can learn.

```python
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out
```

**Answer 14: Multi-Head Attention**

Multiple heads allow the model to attend to information from different representation subspaces at different positions simultaneously. One head might learn to look for consonants, another for positional patterns, another for syntactic relationships. Each head operates on a smaller dimension (`n_embd // n_head`), and their outputs are concatenated and projected back to `n_embd`. This is analogous to group convolutions in CNNs. The projection layer is the "way back" into the residual pathway.

```python
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
```

---

### Part 6: Feed-Forward Network

**Answer 15: Adding a Feed-Forward Layer**

After tokens communicate via self-attention, they need to individually process the information they've gathered. The feed-forward network is applied identically and independently to each position — it's a per-token MLP. The 4× expansion in the inner layer (e.g., 384 → 1536 → 384) follows the original Transformer paper and gives the network more capacity for per-token computation. The ReLU introduces nonlinearity.

```python
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)
```

---

### Part 7: Transformer Block, Residual Connections, and Layer Norm

**Answer 16: The Transformer Block**

The block is the fundamental repeating unit of the Transformer. It interleaves communication (multi-head self-attention) with computation (feed-forward), forming a pattern: attend → compute → attend → compute, stacked `n_layer` times. This is the architecture that, when scaled up, powers GPT.

```python
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x
```

**Answer 17: Residual (Skip) Connections**

Residual connections create a "gradient superhighway": during backpropagation, the addition node distributes gradients equally to both branches (the residual pathway and the transformation block). This means the supervision signal from the loss can flow all the way back to the input without being diminished by the depth of the network. The blocks are initialized to contribute very little at first (their outputs are near zero), so the network starts as a near-identity function and the blocks gradually "come online" during optimization.

```python
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(x)    # residual connection around attention
        x = x + self.ffwd(x)  # residual connection around feed-forward
        return x
```

Also add the projection + dropout in multi-head attention and feed-forward to project back into the residual pathway:

```python
# In MultiHeadAttention:
self.proj = nn.Linear(head_size * num_heads, n_embd)
self.dropout = nn.Dropout(dropout)
# ...
out = self.dropout(self.proj(out))

# In FeedForward:
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd),
    nn.Dropout(dropout),
)
```

**Answer 18: Layer Normalization**

LayerNorm normalizes each token's feature vector to zero mean and unit variance across the `n_embd` dimensions. Unlike BatchNorm (which normalizes across the batch and requires running statistics), LayerNorm operates per-example and has no train/test distinction. This makes it simpler and avoids coupling across batch elements.

We use the **pre-norm** formulation: apply LayerNorm *before* the attention and feed-forward, not after. This is a deviation from the original 2017 paper but has become standard practice because it leads to better training dynamics.

```python
class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
```

Add a final LayerNorm in the model before the lm_head:
```python
self.ln_f = nn.LayerNorm(n_embd)
# in forward:
x = self.ln_f(x)
logits = self.lm_head(x)
```

---

### Part 8: Dropout and Scaling Up

**Answer 19: Adding Dropout**

Dropout is a regularization technique from a 2014 paper. During each forward pass, it randomly zeroes out a fraction of activations (here, 20%). This trains an implicit ensemble of sub-networks — each forward pass uses a different random subset of the network. At test time, all activations are used, effectively averaging the ensemble. We add dropout at three points: after softmax in attention (randomly prevents some token-to-token communication), after the projection in multi-head attention, and in the feed-forward network.

```python
# In Head.__init__:
self.dropout = nn.Dropout(dropout)

# In Head.forward, after softmax:
wei = self.dropout(wei)

# In MultiHeadAttention.forward:
out = self.dropout(self.proj(out))

# In FeedForward (already shown above):
nn.Dropout(dropout)  # after the second Linear
```

**Answer 20: Scaling Up and Final Training**

The final model has ~10.7M parameters. The architecture is identical to what we built; only the hyperparameters change. With `block_size=256`, the model can attend to 256 characters of context. The 6 heads of 64-dimensional attention (384/6=64) and 6 layers of blocks give the model enough depth and width to learn complex patterns. The validation loss of ~1.48 is a dramatic improvement from the bigram model's ~2.5.

```python
# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```

---

### Part 9: Encoder vs. Decoder and the Road to ChatGPT

**Answer 21: Understanding the Full Transformer Architecture**

1. **Decoder-only**: Our model uses the triangular causal mask (`tril`) in attention, preventing future tokens from influencing past predictions. This autoregressive property is what makes it a "decoder" — it can generate text sequentially, one token at a time. There is no encoder and no cross-attention block.

2. **Encoder block**: Identical to what we built, except without the causal mask. All tokens attend to all other tokens. This is useful when the entire input is available upfront (classification, encoding a sentence for translation).

3. **Cross-attention**: In the original Transformer's encoder-decoder architecture (for machine translation), the decoder has an extra attention layer where queries come from the decoder's tokens, but keys and values come from the encoder's output. This lets the decoder "look at" the encoded source sentence while generating the translation.

4. **From our model to ChatGPT**: The architecture is essentially the same decoder-only Transformer. The differences are scale (GPT-3 has 175B parameters vs. our 10M; trained on 300B tokens vs. our ~300K) and the fine-tuning pipeline. After pre-training (which produces a document completer that babbles internet text), ChatGPT undergoes:
   - **Supervised Fine-Tuning (SFT)**: trained on curated question-answer pairs to learn the assistant format.
   - **Reward Model Training**: human raters rank different model responses; a reward model is trained to predict these preferences.
   - **RLHF (PPO)**: the model's generation policy is optimized to maximize the reward model's scores, aligning it to be helpful, harmless, and honest.

The pre-training stage is what nanoGPT (and our exercise) covers. The fine-tuning stages are separate and require proprietary data.

---

### Part 10: Going Further (Karpathy's Challenges)

**Answer 22: Batched Multi-Head Attention**

The key idea is to treat the head dimension as a batch dimension. Instead of creating `n_head` separate `nn.Linear` layers, we use a single linear projection for all heads and reshape the output to `(B, n_head, T, head_size)`. The `@` operator then broadcasts over both the batch *and* head dimensions in parallel. This is dramatically faster on a GPU because it replaces a Python loop with a single batched matrix multiply.

```python
class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        # key, query, value projections for all heads, in a single batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # causal mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        # compute q, k, v for all heads in one go: (B, T, 3*n_embd)
        q, k, v = self.c_attn(x).split(C, dim=2)   # each is (B, T, n_embd)

        # reshape into (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)

        # attention: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        # weighted aggregation: (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        out = wei @ v

        # reassemble heads: (B, T, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.c_proj(out))
        return out
```

The `.transpose(1, 2)` swaps the time and head dimensions so the `T` dimension is next to `head_size`, making the matmul over `(T, hs)` slices natural. At the end, `.contiguous().view(B, T, C)` flattens the heads back into a single channel dimension. This is exactly the pattern used in nanoGPT's `model.py`.

**Answer 23: Train on Your Own Dataset**

Here is a skeleton for the addition challenge. The key changes are: (1) a custom data loader that generates random addition problems, (2) reversed output digits, and (3) masking the loss on the input portion.

```python
import random

def generate_addition_example(max_digits=3):
    """Generate one 'a+b=c' example with reversed answer digits."""
    a = random.randint(0, 10**max_digits - 1)
    b = random.randint(0, 10**max_digits - 1)
    c = a + b
    # e.g. "123+456=975" but reverse the answer: "123+456=579"
    prompt = f"{a}+{b}="
    answer = str(c)[::-1]  # reverse the digits
    return prompt, answer

# Build a character-level vocabulary for digits + operators
chars = sorted(list(set('0123456789+=\n')))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

def get_batch_addition(batch_size, block_size, max_digits=3):
    """Serve a batch of random addition problems."""
    xs, ys = [], []
    for _ in range(batch_size):
        prompt, answer = generate_addition_example(max_digits)
        full = prompt + answer + '\n'
        tokens = encode(full)
        # Pad or truncate to block_size
        if len(tokens) > block_size + 1:
            tokens = tokens[:block_size + 1]
        while len(tokens) < block_size + 1:
            tokens.append(stoi['\n'])  # pad with newlines

        x = tokens[:-1]
        y = tokens[1:]

        # Mask out loss on the prompt portion (everything before and including '=')
        eq_pos = full.index('=')  # position of '='
        for i in range(eq_pos):
            y[i] = -1  # will be ignored by CrossEntropyLoss

        xs.append(x)
        ys.append(y)

    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
    )

# In training, use ignore_index=-1:
# loss = F.cross_entropy(logits, targets, ignore_index=-1)
```

For other datasets, simply replace `input.txt` with your data and adjust the tokenizer. The rest of the model code stays the same. Fun datasets to try: nursery rhymes, cooking recipes, Python source code, or Wikipedia articles.

**Answer 24: Pretraining and Fine-Tuning**

The idea is transfer learning: a model pretrained on a large corpus has already learned general language patterns (character co-occurrence, word boundaries, syntactic structure). Fine-tuning on tiny Shakespeare then only needs to learn the stylistic specifics.

```python
import os

# ---- Step 1: Pretrain on a large dataset ----
# Use e.g. OpenWebText, a large book corpus, or concatenated Project Gutenberg texts.
# The dataset should be large enough that train_loss ≈ val_loss (no overfitting).

# ... (same model definition as before, potentially with a larger block_size) ...

# After pretraining:
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'iter': iter,
    'val_loss': losses['val'],
}, 'pretrained_checkpoint.pt')

# ---- Step 2: Fine-tune on tiny Shakespeare ----
# Load the pretrained checkpoint
checkpoint = torch.load('pretrained_checkpoint.pt')
model = GPTLanguageModel()  # same architecture
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

# Use a lower learning rate and fewer iterations for fine-tuning
finetune_lr = 1e-4          # 3-10x smaller than pretraining lr
finetune_iters = 1000       # much fewer steps
optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)

# Load tiny Shakespeare as before
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
# ... (same tokenization, train/val split, get_batch) ...

for iter in range(finetune_iters):
    if iter % 100 == 0:
        losses = estimate_loss()
        print(f"ft step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Compare this val_loss against a model trained from scratch on tiny Shakespeare.
# If pretraining helped, the fine-tuned val_loss should be lower.
```

A good choice for the pretraining corpus is the full Project Gutenberg collection (~3GB of English literature). With a 10M-parameter model, this is large enough to avoid overfitting. You should see a noticeable improvement in validation loss on tiny Shakespeare after fine-tuning vs. training from scratch, especially when the fine-tuning budget is small.

**Answer 25: Read and Implement a Transformer Improvement**

Here are starter implementations for several popular improvements:

**Option A — Weight Tying** (simplest, often gives ~0.02 loss improvement):
```python
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Tie weights: lm_head shares the token embedding matrix
        self.lm_head.weight = self.token_embedding_table.weight
```
This reduces the parameter count and acts as a regularizer — the model must learn embeddings that are useful for both input and output.

**Option B — Cosine Learning Rate Schedule with Warmup**:
```python
import math

warmup_iters = 100
lr_decay_iters = max_iters
min_lr = learning_rate / 10  # 3e-5

def get_lr(it):
    # linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # cosine decay
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# In the training loop:
for iter in range(max_iters):
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # ... rest of training step ...
```

**Option C — RMSNorm** (replacing LayerNorm):
```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

# Replace nn.LayerNorm(n_embd) with RMSNorm(n_embd) everywhere
```

**Option D — SwiGLU Feed-Forward** (replacing ReLU FFN):
```python
class SwiGLUFeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        # SwiGLU uses 8/3 * n_embd to keep param count ≈ same as 4 * n_embd with ReLU
        hidden = int(8 / 3 * n_embd)
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)  # gate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

To properly evaluate, train two identical models (same seed, same data, same number of iterations) — one with the change and one without — and compare validation loss curves. Weight tying and cosine LR are the easiest wins; RoPE and SwiGLU require more careful implementation but are used in all modern LLMs (LLaMA, Mistral, etc.).
