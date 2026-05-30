## Questions

These exercises follow the chronological flow of the **makemore Part 4: Becoming a Backprop Ninja** video and the notebook [makemore/makemore_part4_backprop.ipynb](makemore/makemore_part4_backprop.ipynb). Work through them in order to manually implement the full backward pass of a two-layer MLP with batch normalization.

---

### Part 1: Setup

**Exercise 1: Imports, Data, and Utility Function**  
Set up the environment:
- Import `torch`, `torch.nn.functional as F`, and `matplotlib.pyplot as plt`.
- Load and process the dataset: vocabulary, `stoi`/`itos` mappings, `vocab_size`, and the three dataset splits.
- Write a `cmp(s, dt, t)` utility function that:
  - Checks whether a manually computed gradient tensor `dt` matches `t.grad` from PyTorch exactly (`torch.all`) and approximately (`torch.allclose`).
  - Computes and prints the maximum absolute difference `(dt - t.grad).abs().max().item()`.
  - Prints a one-line summary: `s | exact: ... | approximate: ... | maxdiff: ...`.

**Exercise 2: Model Initialization and Chunked Forward Pass**  
Initialize the two-layer MLP with batch normalization:
- Use `n_embd = 10`, `n_hidden = 64`.
- Initialize all parameters with a fixed `Generator(seed=2147483647)`. Use **small random numbers** (not zeros) for biases and batch-norm parameters to avoid masking gradient bugs (e.g. `b1 = randn * 0.1`, `bngain = randn * 0.1 + 1.0`, `bnbias = randn * 0.1`).
- Include `b1` even though it is redundant after batch normalization (useful for verifying its gradient is correct).
- Sample a single mini-batch of size 32.
- Implement the forward pass as a sequence of **named intermediate tensors**, breaking the computation into atoms:
  - Embedding lookup → `emb`, `embcat`
  - First linear layer → `hprebn`
  - Batch norm internals → `bnmeani`, `bndiff`, `bndiff2`, `bnvar` (with Bessel's correction: `1/(n-1)`), `bnvar_inv`, `bnraw`, `hpreact`
  - Non-linearity → `h`
  - Second linear layer → `logits`
  - Loss internals → `logit_maxes`, `norm_logits`, `counts`, `counts_sum`, `counts_sum_inv`, `probs`, `logprobs`, `loss`
- Call `retain_grad()` on each intermediate tensor and run `loss.backward()` so you can later compare against PyTorch's gradients.

---

### Part 2: Manual Backpropagation — Atomic Steps

**Exercise 3: Backward Through the Loss**  
Manually derive and implement the gradients for each intermediate tensor of the loss, in reverse order:  
`dlogprobs → dprobs → dcounts_sum_inv → dcounts_sum → dcounts → dnorm_logits → dlogit_maxes → dlogits (first contribution)`  

Key points:
- `dlogprobs`: shape is `(n, vocab_size)`. Only `n` elements participate in the loss; all others have gradient 0.
- Broadcasting: wherever the forward pass broadcast a tensor (e.g. `counts_sum_inv`), the backward pass must **sum** over the replicated dimension.
- Node reuse: `counts` is used in two places; accumulate both contributions with `+=`.
- After computing `dlogit_maxes`, inspect its values and explain why they are near zero.

**Exercise 4: Backward Through the Second Linear Layer**  
Back-propagate through `logits = h @ W2 + b2`:
- Recover `dh`, `dW2`, `db2` using matrix-multiply rules (derive shapes first, then determine which transposition makes dimensions match).
- Add the second contribution to `dlogits` (through the `logit_maxes` branch).
- Use `cmp` to verify all three gradients.

**Exercise 5: Backward Through Tanh and Batch Norm Parameters**  
Back-propagate through:
- `h = tanh(hpreact)` → `dhpreact` using the formula `(1 - h**2) * dh`.
- `hpreact = bngain * bnraw + bnbias` → `dbngain`, `dbnraw`, `dbnbias`, taking care that `bngain` and `bnbias` are shape `(1, n_hidden)` and must be summed over the batch dimension (keep dims).
- Use `cmp` to verify all three.

**Exercise 6: Backward Through the Batch Norm Internals**  
Back-propagate step by step through:  
`bnraw → bndiff / bnvar_inv → bnvar → bndiff2 → bndiff (second contribution) → bnmeani → hprebn`  

Notes:
- `bnvar_inv = (bnvar + 1e-5)**-0.5`: apply the power rule.
- Bessel's correction: the factor in `dbnvar → dbndiff2` is `1/(n-1)`.
- `bndiff` is used in two branches; accumulate both with `+=`.
- When back-propagating through `bnmeani` into `hprebn`, note that the gradient copies and also routes a 1/n-scaled version back.

**Exercise 7: Backward Through the First Linear Layer and Embedding**  
Back-propagate through:
- `hprebn = embcat @ W1 + b1` → `dembcat`, `dW1`, `db1` (same shape-matching strategy as Exercise 4).
- `embcat = emb.view(...)` → `demb` by re-viewing `dembcat` into the original embedding shape.
- Embedding lookup `emb = C[Xb]` → `dC` using a double for-loop: for each `(k, j)` position, accumulate `demb[k, j]` into `dC[Xb[k, j]]` with `+=`.
- Use `cmp` to verify all gradients.

---

### Part 3: Efficient Backward Passes

**Exercise 8: Fused Cross-Entropy Backward**  
Replace the multi-step loss computation with a single `F.cross_entropy(logits, Yb)` call and derive `dlogits` analytically in one shot:
- The gradient of the cross-entropy loss with respect to logit `i` of example `e` is:
  - `softmax(logits)[e, i]` if `i ≠ label[e]`
  - `softmax(logits)[e, i] - 1` if `i == label[e]`
  - All divided by `n` (because the loss is a mean).
- Implement this, verify with `cmp`, and visualize `dlogits` as a `(32, 27)` grayscale image — white squares should appear at the correct label positions.

**Exercise 9: Fused Batch Norm Backward**  
Replace the step-by-step batch norm backward with a single analytical formula for `dhprebn` given `dhpreact`:

$$\frac{\partial L}{\partial x_i} = \frac{\gamma \cdot \hat{\sigma}^{-1}}{n} \left( n \cdot \frac{\partial L}{\partial y_i} - \sum_j \frac{\partial L}{\partial y_j} - \frac{n}{n-1} \hat{x}_i \sum_j \frac{\partial L}{\partial y_j} \hat{x}_j \right)$$

where $\hat{\sigma}^{-1}$ = `bnvar_inv` and $\hat{x}$ = `bnraw`.

Implement this as a single line of code and verify with `cmp`.

---

### Part 4: Full Manual Training Loop

**Exercise 10: Train with Manual Backward Pass**  
Assemble the complete training loop using your own gradients instead of `loss.backward()`:
- Re-initialize the network with `n_embd = 10`, `n_hidden = 200`.
- In the training loop:
  - Use the fused forward pass (`F.cross_entropy`, then the analytic `dlogits`).
  - Chain through: second layer → tanh → fused batch norm backward → first layer → embedding.
  - Wrap the entire loop in `with torch.no_grad()` since no autograd graph is needed.
  - Update parameters using the manually computed gradients (not `p.grad`).
- After training, calibrate batch norm statistics over the full training set in a separate `torch.no_grad()` pass.
- Evaluate train and val loss with `split_loss`. You should reach approximately 2.07 / 2.12.
- Sample 20 names from the model to visually confirm it works.

---

## Answers

### Part 1: Setup

**Exercise 1**

The `cmp` utility is the diagnostic backbone of this lecture. Because we are writing gradients by hand, we need a reliable way to check that they match PyTorch's autograd. `torch.all` checks bitwise equality while `torch.allclose` tolerates small floating-point rounding; `maxdiff` quantifies the worst-case error so you can distinguish a correct-but-noisy result from a wrong one.

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline

def cmp(s, dt, t):
    ex      = torch.all(dt == t.grad).item()
    app     = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
```

**Exercise 2**

Parameters are initialised with small random numbers instead of zeros deliberately. When a parameter is exactly zero, the gradient expression simplifies and can hide bugs where two wrong terms cancel. Small non-zero values keep the full expression active so errors show up immediately. The bias `b1` is kept despite the following batch-norm layer (which subtracts the mean and therefore makes `b1` useless) purely to verify that its gradient is correctly computed as approximately zero.

Bessel's correction (`1/(n-1)`) is used instead of the paper's `1/n` because the mini-batch is a small sample of the population; dividing by `n` systematically underestimates the variance, whereas `n-1` gives an unbiased estimate.

```python
n_embd  = 10
n_hidden = 64
g = torch.Generator().manual_seed(2147483647)
C      = torch.randn((vocab_size, n_embd),            generator=g)
W1     = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1     = torch.randn(n_hidden,                        generator=g) * 0.1
W2     = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2     = torch.randn(vocab_size,                      generator=g) * 0.1
bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

batch_size = 32
n  = batch_size
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix]

# forward pass — chunked
emb    = C[Xb]
embcat = emb.view(emb.shape[0], -1)
hprebn = embcat @ W1 + b1

bnmeani   = 1/n * hprebn.sum(0, keepdim=True)
bndiff    = hprebn - bnmeani
bndiff2   = bndiff**2
bnvar     = 1/(n-1) * bndiff2.sum(0, keepdim=True)     # Bessel's correction
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw     = bndiff * bnvar_inv
hpreact   = bngain * bnraw + bnbias

h       = torch.tanh(hpreact)
logits  = h @ W2 + b2

logit_maxes   = logits.max(1, keepdim=True).values
norm_logits   = logits - logit_maxes
counts        = norm_logits.exp()
counts_sum    = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1
probs         = counts * counts_sum_inv
logprobs      = probs.log()
loss          = -logprobs[range(n), Yb].mean()

for t in [logprobs, probs, counts, counts_sum, counts_sum_inv,
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
          bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani, embcat, emb]:
    t.retain_grad()
loss.backward()
```

---

### Part 2: Manual Backpropagation — Atomic Steps

**Exercise 3**

Back-propagation is the repeated application of the chain rule from the loss backwards through each operation. For the loss `loss = -logprobs[range(n), Yb].mean()`, only `n` elements of `logprobs` participated: their gradient is `-1/n`; all others are `0`. Each subsequent gradient is local-derivative × incoming global gradient.

Whenever a forward operation **broadcast** a tensor (replicating values across a dimension), the backward pass must **sum** across that same dimension, because all those replicated copies receive a gradient and they all route back to the same original parameter. Conversely, whenever a forward operation **summed** across a dimension, the backward pass **replicates** the gradient along that dimension.

```python
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n

dprobs = (1.0 / probs) * dlogprobs

dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)

dcounts = counts_sum_inv * dprobs                        # first branch

dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv

dcounts += torch.ones_like(counts) * dcounts_sum         # second branch (+=)

dnorm_logits = counts * dcounts

dlogits = dnorm_logits.clone()                           # first contribution

dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
# logit_maxes.grad is near zero — subtracting the max is numerically
# stabilising but does not change softmax output, so the gradient vanishes.
```

**Exercise 4**

For a matrix multiplication `D = A @ B + C`, the gradients are `dA = dD @ B.T`, `dB = A.T @ dD`, `dC = dD.sum(0)`. There is no need to memorise these; just ensure the shapes work out — there is exactly one transposition arrangement that produces the correct output shape.

```python
dh   = dlogits @ W2.T
dW2  = h.T @ dlogits
db2  = dlogits.sum(0)

# second contribution to dlogits from the logit_maxes branch
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes

cmp('h',  dh,  h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)
```

**Exercise 5**

The local gradient of `tanh` at output value `t` is `1 - t²`. When `|t| ≈ 1` (saturated), this gradient is near zero — gradients stop flowing and neurons stop learning. Batch-norm parameters `bngain` and `bnbias` are shape `(1, n_hidden)` but operate on a `(n, n_hidden)` tensor; they are replicated across the batch in the forward pass, so their gradients must be summed over the batch dimension and `keepdim=True` must be used to preserve the leading `1`.

```python
dhpreact = (1.0 - h**2) * dh

dbngain = (bnraw  * dhpreact).sum(0, keepdim=True)
dbnraw  =  bngain * dhpreact
dbnbias =           dhpreact.sum(0, keepdim=True)

cmp('bngain', dbngain, bngain)
cmp('bnbias', dbnbias, bnbias)
cmp('bnraw',  dbnraw,  bnraw)
```

**Exercise 6**

The batch-norm backward is the most intricate part. Each intermediate node in the batch-norm graph must be differentiated in topological order. Two key facts drive the complexity: (1) `bndiff` is used in *two* places (`bnraw` and `bndiff2`), so its gradient accumulates from both branches; (2) `bnmeani` is computed from `hprebn`, which means `hprebn` also receives a gradient through the mean subtraction path.

```python
dbndiff  = bnvar_inv * dbnraw                                          # from bnraw
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)

dbnvar   = (-0.5 * (bnvar + 1e-5)**-1.5) * dbnvar_inv

dbndiff2 = (1.0/(n-1)) * torch.ones_like(bndiff2) * dbnvar

dbndiff += (2 * bndiff) * dbndiff2                                     # second branch +=

dhprebn  = dbndiff.clone()
dbnmeani = (-dbndiff).sum(0)
dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)

cmp('hprebn',  dhprebn,  hprebn)
cmp('bnmeani', dbnmeani, bnmeani)
```

**Exercise 7**

The embedding lookup is the reverse of a scatter: for every `(k, j)` position in the batch, the gradient `demb[k, j]` must be accumulated (`+=`) into the row `dC[Xb[k, j]]`. The `+=` is essential because the same row of `C` (character embedding) may have been used by many examples in the batch.

```python
dembcat = dhprebn @ W1.T
dW1     = embcat.T @ dhprebn
db1     = dhprebn.sum(0)

demb = dembcat.view(emb.shape)

dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]

cmp('W1',     dW1,     W1)
cmp('b1',     db1,     b1)
cmp('embcat', dembcat, embcat)
cmp('emb',    demb,    emb)
cmp('C',      dC,      C)
```

---

### Part 3: Efficient Backward Passes

**Exercise 8**

Differentiating the full softmax + negative log-likelihood analytically yields a remarkably clean result: the gradient of the loss with respect to logit `(e, i)` is simply `softmax(logits)[e, i]` everywhere, minus 1 at the correct label position, divided by `n`. This collapses dozens of backward steps into three lines. Intuitively, for each example the gradient acts as a force: it pulls down the probabilities of incorrect characters and pulls up the probability of the correct one, with the magnitude of the pull proportional to the current predicted probability.

```python
loss_fast = F.cross_entropy(logits, Yb)

dlogits = F.softmax(logits, 1)
dlogits[range(n), Yb] -= 1
dlogits /= n

cmp('logits', dlogits, logits)   # approximate=True, maxdiff ≈ 6e-9

plt.figure(figsize=(4, 4))
plt.imshow(dlogits.detach(), cmap='gray')
plt.title('dlogits — black squares at correct labels')
```

**Exercise 9**

Analytically collapsing the entire batch-norm backward into a single expression removes roughly 10 intermediate backward steps. The derivation requires careful bookkeeping: `bnvar` and `bnmeani` are scalars (one per neuron), so they receive gradient contributions from all `n` examples. When MU is defined as the batch mean (as it is here), the `d(sigma²)/d(mu)` term vanishes, simplifying the expression considerably. The final formula runs in parallel over all `n_hidden` neurons via broadcasting.

```python
# given dhpreact, bngain, bnvar_inv (= (bnvar+1e-5)^-0.5), bnraw, n
dhprebn = bngain * bnvar_inv / n * (
    n * dhpreact
    - dhpreact.sum(0)
    - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0)
)

cmp('hprebn', dhprebn, hprebn)   # approximate=True, maxdiff ≈ 9e-10
```

---

### Part 4: Full Manual Training Loop

**Exercise 10**

Wrapping the entire loop in `with torch.no_grad()` tells PyTorch that no computational graph needs to be built. This is correct because we are computing all gradients ourselves; we are only using PyTorch as an efficient tensor engine. The manual gradients are arranged in the same order as `parameters`, so they can be zipped and applied directly without accessing `.grad`.

```python
n_embd   = 10
n_hidden = 200
g = torch.Generator().manual_seed(2147483647)
C      = torch.randn((vocab_size, n_embd),            generator=g)
W1     = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1     = torch.randn(n_hidden,                        generator=g) * 0.1
W2     = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2     = torch.randn(vocab_size,                      generator=g) * 0.1
bngain = torch.randn((1, n_hidden), generator=g) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden), generator=g) * 0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True

max_steps  = 200000
batch_size = 32
n = batch_size
lossi = []

with torch.no_grad():
    for i in range(max_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        # forward
        emb       = C[Xb]
        embcat    = emb.view(emb.shape[0], -1)
        hprebn    = embcat @ W1 + b1
        bnmean    = hprebn.mean(0, keepdim=True)
        bnvar     = hprebn.var(0,  keepdim=True, unbiased=True)
        bnvar_inv = (bnvar + 1e-5)**-0.5
        bnraw     = (hprebn - bnmean) * bnvar_inv
        hpreact   = bngain * bnraw + bnbias
        h         = torch.tanh(hpreact)
        logits    = h @ W2 + b2
        loss      = F.cross_entropy(logits, Yb)

        # manual backward
        dlogits   = F.softmax(logits, 1); dlogits[range(n), Yb] -= 1; dlogits /= n
        dh        = dlogits @ W2.T
        dW2       = h.T @ dlogits
        db2       = dlogits.sum(0)
        dhpreact  = (1.0 - h**2) * dh
        dbngain   = (bnraw  * dhpreact).sum(0, keepdim=True)
        dbnbias   =           dhpreact.sum(0, keepdim=True)
        dhprebn   = bngain * bnvar_inv / n * (
                        n * dhpreact - dhpreact.sum(0)
                        - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0))
        dembcat   = dhprebn @ W1.T
        dW1       = embcat.T @ dhprebn
        db1       = dhprebn.sum(0)
        demb      = dembcat.view(emb.shape)
        dC        = torch.zeros_like(C)
        for k in range(Xb.shape[0]):
            for j in range(Xb.shape[1]):
                ix2 = Xb[k, j]
                dC[ix2] += demb[k, j]
        grads = [dC, dW1, db1, dW2, db2, dbngain, dbnbias]

        lr = 0.1 if i < 100000 else 0.01
        for p, grad in zip(parameters, grads):
            p.data += -lr * grad

        if i % 10000 == 0:
            print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
        lossi.append(loss.log10().item())

# calibrate batch norm
with torch.no_grad():
    emb     = C[Xtr]
    embcat  = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    bnmean  = hpreact.mean(0, keepdim=True)
    bnvar   = hpreact.var(0,  keepdim=True, unbiased=True)

@torch.no_grad()
def split_loss(split):
    x, y = {'train': (Xtr, Ytr), 'val': (Xdev, Ydev), 'test': (Xte, Yte)}[split]
    emb     = C[x]
    embcat  = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1 + b1
    hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
    h       = torch.tanh(hpreact)
    logits  = h @ W2 + b2
    loss    = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')   # ≈ 2.07
split_loss('val')     # ≈ 2.12
```
