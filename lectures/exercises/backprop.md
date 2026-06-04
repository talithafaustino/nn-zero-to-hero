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

**`dlogprobs`** — Forward: `loss = -logprobs[range(n), Yb].mean()`. The `logprobs` tensor has shape `(n, vocab_size)` = `(32, 27)`, but only 32 elements participate in the loss — one per row, at the column index given by `Yb`. The `.mean()` divides by `n`, and the `-` negates. So for each selected element, `∂loss/∂logprobs[e, Yb[e]] = -1/n`. All other elements have gradient `0` because they don't appear in the loss expression at all. We create a zeros tensor of the same shape and fill in `-1/n` at the 32 correct positions.

**`dprobs`** — Forward: `logprobs = probs.log()`. This is element-wise `log`. The derivative of `log(x)` is `1/x`. So the local derivative is `1/probs`, shape `(32, 27)`. By the chain rule we multiply element-wise: `dprobs = (1/probs) * dlogprobs`. Shape stays `(32, 27)`. Most entries are zero because `dlogprobs` is zero there.

**`dcounts_sum_inv`** — Forward: `probs = counts * counts_sum_inv`. Here `counts` is `(32, 27)` and `counts_sum_inv` is `(32, 1)`. PyTorch broadcasts `counts_sum_inv` across the 27 columns. For the derivative w.r.t. `counts_sum_inv`: the local derivative of `a*b` w.r.t. `b` is `a` = `counts`. So the raw gradient is `counts * dprobs`, shape `(32, 27)`. But `counts_sum_inv` is `(32, 1)` — it was broadcast across 27 columns in the forward pass, so in the backward pass we must **sum** over the column dimension (dim=1) to collapse back to `(32, 1)`. Hence `.sum(1, keepdim=True)`.

**`dcounts` (first branch)** — Forward: `probs = counts * counts_sum_inv`. The local derivative of `a*b` w.r.t. `a` is `b` = `counts_sum_inv`. So `dcounts = counts_sum_inv * dprobs`. Here `counts_sum_inv` is `(32, 1)` and `dprobs` is `(32, 27)` — PyTorch broadcasts `counts_sum_inv` to `(32, 27)`, giving `dcounts` shape `(32, 27)`. This is only the **first** gradient contribution to `counts`; `counts` is also used inside `counts_sum`, so we'll add a second contribution later.

**`dcounts_sum`** — Forward: `counts_sum_inv = counts_sum**-1`. This is `f(x) = x⁻¹`, whose derivative is `-x⁻²`. So the local derivative is `-counts_sum**-2`, shape `(32, 1)`. Chain rule: `dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv`. Both tensors are `(32, 1)`, so the result is `(32, 1)`.

**`dcounts += ...` (second branch)** — Forward: `counts_sum = counts.sum(1, keepdims=True)`. Summing over dim=1 collapses `(32, 27)` → `(32, 1)`. Backward: summing is a "fan-in" operation; its reverse is "fan-out" — the gradient is simply copied (replicated) back to each of the 27 positions that were summed. So `dcounts_sum` (shape `(32, 1)`) is broadcast across 27 columns: `torch.ones_like(counts) * dcounts_sum` gives shape `(32, 27)`. We use `+=` because `counts` has already received a gradient from the `probs` branch above.

**`dnorm_logits`** — Forward: `counts = norm_logits.exp()`. The derivative of `eˣ` is `eˣ` itself, which is just `counts`. So `dnorm_logits = counts * dcounts`, element-wise multiply, shape `(32, 27)`.

**`dlogits` (first contribution)** — Forward: `norm_logits = logits - logit_maxes`. The derivative of `a - b` w.r.t. `a` is `+1`. So the first contribution is simply `dlogits = dnorm_logits.clone()`, shape `(32, 27)`. We use `.clone()` because we'll add a second contribution later.

**`dlogit_maxes`** — Forward: `norm_logits = logits - logit_maxes`. The derivative w.r.t. `logit_maxes` is `-1`. `logit_maxes` is `(32, 1)` and was broadcast across 27 columns, so in the backward pass we sum over dim=1: `dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)`, shape `(32, 1)`. This value is very close to zero: subtracting the max is a numerically stabilising trick that doesn't change the softmax output, so the gradient through it effectively vanishes. Each row of `dnorm_logits` sums to approximately zero because the softmax probabilities sum to 1.

```python
# dlogprobs: (32, 27) — only the 32 correct-label positions get -1/n
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n

# dprobs: (32, 27) — local deriv of log(x) is 1/x, chain rule * dlogprobs
dprobs = (1.0 / probs) * dlogprobs

# dcounts_sum_inv: (32, 1) — deriv of counts*counts_sum_inv w.r.t. counts_sum_inv
# is counts; sum over dim=1 because counts_sum_inv was broadcast across 27 cols
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)

# dcounts (1st branch): (32, 27) — deriv of counts*counts_sum_inv w.r.t. counts
# is counts_sum_inv; it broadcasts from (32,1) to (32,27)
dcounts = counts_sum_inv * dprobs

# dcounts_sum: (32, 1) — deriv of x^-1 is -x^-2
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv

# dcounts (2nd branch): sum was a fan-in, so backward fans-out (replicates)
# dcounts_sum (32,1) broadcasts to (32,27); += accumulates both branches
dcounts += torch.ones_like(counts) * dcounts_sum

# dnorm_logits: (32, 27) — deriv of exp(x) is exp(x) = counts
dnorm_logits = counts * dcounts

# dlogits (1st contribution): (32, 27) — deriv of (logits - logit_maxes) w.r.t. logits is +1
dlogits = dnorm_logits.clone()

# dlogit_maxes: (32, 1) — deriv w.r.t. logit_maxes is -1; sum over cols (was broadcast)
# near zero because subtracting the max doesn't change the softmax output
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
```

**Exercise 4**

For a matrix multiplication `D = A @ B + C`, the gradients are `dA = dD @ B.T`, `dB = A.T @ dD`, `dC = dD.sum(0)`. There is no need to memorise these; just ensure the shapes work out — there is exactly one transposition arrangement that produces the correct output shape.

**`dh`** — Forward: `logits = h @ W2 + b2`, where `h` is `(32, 64)`, `W2` is `(64, 27)`, `logits` is `(32, 27)`. To get the gradient w.r.t. `h`, we need `dlogits @ W2.T`. Dimension check: `(32, 27) @ (27, 64)` → `(32, 64)` ✓ — matches `h`'s shape. Intuitively, each row of `dh` is the gradient on logits "pulled back" through `W2` to the hidden layer.

**`dW2`** — To get the gradient w.r.t. `W2`, we need `h.T @ dlogits`. Dimension check: `(64, 32) @ (32, 27)` → `(64, 27)` ✓ — matches `W2`'s shape. Each element `dW2[i,j]` accumulates over the batch how much neuron `i`'s activation contributed to logit `j`.

**`db2`** — `b2` has shape `(27,)` and is broadcast (added) to every row of `h @ W2`. Backward through a broadcast-add means sum over the batch dimension: `db2 = dlogits.sum(0)`, collapsing `(32, 27)` → `(27,)`.

**`dlogits +=` (second contribution)** — `logits` also feeds into `logit_maxes = logits.max(1)`. The `max` operation selects one element per row; only that element receives gradient. `F.one_hot(logits.max(1).indices, ...)` creates a `(32, 27)` mask with a `1` at the max position in each row. Multiplying by `dlogit_maxes` (which is `(32, 1)`, broadcast to `(32, 27)`) gives the second gradient contribution. We use `+=` to accumulate onto `dlogits`.

```python
# dh: (32, 64) — pull dlogits back through W2
dh   = dlogits @ W2.T
# dW2: (64, 27) — outer product accumulated over batch
dW2  = h.T @ dlogits
# db2: (27,) — sum over batch (undo the broadcast-add)
db2  = dlogits.sum(0)

# second contribution to dlogits from the logit_maxes branch
# one_hot mask picks the max element per row; dlogit_maxes broadcasts from (32,1)
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes

cmp('h',  dh,  h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)
```

**Exercise 5**

The local gradient of `tanh` at output value `t` is `1 - t²`. When `|t| ≈ 1` (saturated), this gradient is near zero — gradients stop flowing and neurons stop learning. Batch-norm parameters `bngain` and `bnbias` are shape `(1, n_hidden)` but operate on a `(n, n_hidden)` tensor; they are replicated across the batch in the forward pass, so their gradients must be summed over the batch dimension and `keepdim=True` must be used to preserve the leading `1`.

**`dhpreact`** — Forward: `h = torch.tanh(hpreact)`, element-wise. Both have shape `(32, 64)`. The derivative of `tanh(x)` is `1 - tanh²(x)`, but we already have `h = tanh(hpreact)`, so the local derivative is `1 - h**2`. Chain rule: `dhpreact = (1 - h**2) * dh`, element-wise, shape `(32, 64)`. Where `h` is close to ±1 (saturated neuron), `1 - h²` ≈ 0 and the gradient is killed — this is the "vanishing gradient" problem of tanh.

**`dbngain`** — Forward: `hpreact = bngain * bnraw + bnbias`. `bngain` has shape `(1, 64)` and `bnraw` has shape `(32, 64)`. PyTorch broadcasts `bngain` across the 32 rows. The local derivative of `bngain * bnraw` w.r.t. `bngain` is `bnraw`. So the raw gradient is `bnraw * dhpreact`, shape `(32, 64)`. Since `bngain` is `(1, 64)` (was broadcast over batch), we must sum over dim=0 (batch) to collapse back: `.sum(0, keepdim=True)` → `(1, 64)`.

**`dbnraw`** — Same forward: `hpreact = bngain * bnraw + bnbias`. The local derivative of `bngain * bnraw` w.r.t. `bnraw` is `bngain`. So `dbnraw = bngain * dhpreact`. Here `bngain` is `(1, 64)` and broadcasts to `(32, 64)`, giving `dbnraw` shape `(32, 64)`. No summation needed because `bnraw` is already `(32, 64)`.

**`dbnbias`** — Forward: `hpreact = bngain * bnraw + bnbias`. The local derivative of the `+ bnbias` w.r.t. `bnbias` is `1`. So the raw gradient is just `dhpreact`, shape `(32, 64)`. Since `bnbias` is `(1, 64)` (broadcast over batch), sum over dim=0: `dhpreact.sum(0, keepdim=True)` → `(1, 64)`.

```python
# dhpreact: (32, 64) — tanh derivative is (1 - tanh²), i.e. (1 - h²)
dhpreact = (1.0 - h**2) * dh

# dbngain: (1, 64) — local deriv is bnraw; sum over batch because bngain was broadcast
dbngain = (bnraw  * dhpreact).sum(0, keepdim=True)
# dbnraw: (32, 64) — local deriv is bngain; bngain broadcasts from (1,64) to (32,64)
dbnraw  =  bngain * dhpreact
# dbnbias: (1, 64) — local deriv is 1; sum over batch because bnbias was broadcast
dbnbias =           dhpreact.sum(0, keepdim=True)

cmp('bngain', dbngain, bngain)
cmp('bnbias', dbnbias, bnbias)
cmp('bnraw',  dbnraw,  bnraw)
```

**Exercise 6**

The batch-norm backward is the most intricate part. Each intermediate node in the batch-norm graph must be differentiated in topological order. Two key facts drive the complexity: (1) `bndiff` is used in *two* places (`bnraw` and `bndiff2`), so its gradient accumulates from both branches; (2) `bnmeani` is computed from `hprebn`, which means `hprebn` also receives a gradient through the mean subtraction path.

**`dbndiff` (first branch)** — Forward: `bnraw = bndiff * bnvar_inv`. `bndiff` is `(32, 64)`, `bnvar_inv` is `(1, 64)`. The local derivative of `a * b` w.r.t. `a` is `b`. So `dbndiff = bnvar_inv * dbnraw`. `bnvar_inv` broadcasts from `(1, 64)` to `(32, 64)`. Result shape `(32, 64)`. This is only the first contribution — `bndiff` is also used in `bndiff2`, so we'll add to it later with `+=`.

**`dbnvar_inv`** — Forward: `bnraw = bndiff * bnvar_inv`. The local derivative w.r.t. `bnvar_inv` is `bndiff`. Raw gradient is `bndiff * dbnraw`, shape `(32, 64)`. Since `bnvar_inv` is `(1, 64)` (was broadcast over batch), we sum over dim=0: `.sum(0, keepdim=True)` → `(1, 64)`.

**`dbnvar`** — Forward: `bnvar_inv = (bnvar + 1e-5)**-0.5`. This is `f(x) = (x + ε)^(-0.5)`. By the power rule, `f'(x) = -0.5 * (x + ε)^(-1.5)`. Chain rule: `dbnvar = -0.5 * (bnvar + 1e-5)**-1.5 * dbnvar_inv`. Both tensors are `(1, 64)`, result is `(1, 64)`.

**`dbndiff2`** — Forward: `bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)`. The `.sum(0, keepdim=True)` collapses `(32, 64)` → `(1, 64)`. Backward through a sum replicates the gradient. The factor `1/(n-1)` is Bessel's correction. So `dbndiff2 = 1/(n-1) * ones_like(bndiff2) * dbnvar`. Here `dbnvar` is `(1, 64)` and `torch.ones_like(bndiff2)` is `(32, 64)` — the broadcast copies `dbnvar` into every row. Result shape `(32, 64)`.

**`dbndiff += ...` (second branch)** — Forward: `bndiff2 = bndiff**2`. The derivative of `x²` is `2x`. So `dbndiff += 2 * bndiff * dbndiff2`, element-wise, shape `(32, 64)`. We use `+=` because `bndiff` already received gradient from the `bnraw` branch above. This accumulates the second contribution.

**`dhprebn` (first part)** — Forward: `bndiff = hprebn - bnmeani`. The derivative of `a - b` w.r.t. `a` is `+1`. So the first contribution is `dhprebn = dbndiff.clone()`, shape `(32, 64)`. We clone because we'll add more to `dhprebn` from the `bnmeani` branch.

**`dbnmeani`** — Forward: `bndiff = hprebn - bnmeani`. The derivative w.r.t. `bnmeani` is `-1`. `bnmeani` is `(1, 64)` and was broadcast to `(32, 64)`, so we sum over dim=0: `dbnmeani = (-dbndiff).sum(0)`, shape `(64,)`. Note: this collapses to `(64,)` rather than `(1, 64)` because we don't use `keepdim=True` — it still works because PyTorch will broadcast it back when needed.

**`dhprebn += ...` (second part)** — Forward: `bnmeani = 1/n * hprebn.sum(0, keepdim=True)`. The `.sum(0)` collapses `(32, 64)` → `(1, 64)`, and `1/n` scales it. Backward: the sum replicates the gradient to all 32 rows, and the `1/n` factor stays. So `dhprebn += 1/n * ones_like(hprebn) * dbnmeani`. `dbnmeani` is `(64,)` and broadcasts to `(32, 64)`. This adds the gradient path through the mean to `dhprebn`.

```python
# dbndiff (1st branch): (32, 64) — from bnraw = bndiff * bnvar_inv, deriv w.r.t. bndiff is bnvar_inv
dbndiff  = bnvar_inv * dbnraw
# dbnvar_inv: (1, 64) — deriv w.r.t. bnvar_inv is bndiff; sum over batch (was broadcast)
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)

# dbnvar: (1, 64) — power rule on (x+eps)^(-0.5): deriv is -0.5*(x+eps)^(-1.5)
dbnvar   = (-0.5 * (bnvar + 1e-5)**-1.5) * dbnvar_inv

# dbndiff2: (32, 64) — backward through sum replicates dbnvar to all rows, scaled by 1/(n-1)
dbndiff2 = (1.0/(n-1)) * torch.ones_like(bndiff2) * dbnvar

# dbndiff (2nd branch): (32, 64) — deriv of x² is 2x; += accumulates from both branches
dbndiff += (2 * bndiff) * dbndiff2

# dhprebn (1st part): (32, 64) — from bndiff = hprebn - bnmeani, deriv w.r.t. hprebn is +1
dhprebn  = dbndiff.clone()
# dbnmeani: (64,) — deriv w.r.t. bnmeani is -1; sum over batch (was broadcast from (1,64))
dbnmeani = (-dbndiff).sum(0)
# dhprebn (2nd part): add gradient flowing through bnmeani → hprebn (the mean path, scaled 1/n)
dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)

cmp('hprebn',  dhprebn,  hprebn)
cmp('bnmeani', dbnmeani, bnmeani)
```

**Exercise 7**

The embedding lookup is the reverse of a scatter: for every `(k, j)` position in the batch, the gradient `demb[k, j]` must be accumulated (`+=`) into the row `dC[Xb[k, j]]`. The `+=` is essential because the same row of `C` (character embedding) may have been used by many examples in the batch.

**`dembcat`** — Forward: `hprebn = embcat @ W1 + b1`. `embcat` is `(32, 30)` (= `block_size * n_embd` = `3 * 10`), `W1` is `(30, 64)`, `hprebn` is `(32, 64)`. Same matmul rule as Exercise 4: `dembcat = dhprebn @ W1.T`. Dimension check: `(32, 64) @ (64, 30)` → `(32, 30)` ✓.

**`dW1`** — `dW1 = embcat.T @ dhprebn`. Dimension check: `(30, 32) @ (32, 64)` → `(30, 64)` ✓ — matches `W1`'s shape.

**`db1`** — `b1` is `(64,)`, broadcast-added to every row. Sum over batch: `db1 = dhprebn.sum(0)`, collapsing `(32, 64)` → `(64,)`.

**`demb`** — Forward: `embcat = emb.view(emb.shape[0], -1)`. The `.view()` just reshapes `(32, 3, 10)` → `(32, 30)` without copying data. Backward: we reshape back: `demb = dembcat.view(emb.shape)`, turning `(32, 30)` → `(32, 3, 10)`. No math — just undoing the reshape.

**`dC`** — Forward: `emb = C[Xb]`. This is a lookup table: for each `(k, j)` in `Xb` (shape `(32, 3)`), row `Xb[k,j]` of `C` (shape `(27, 10)`) is copied into `emb[k, j]` (shape `(10,)`). Backward: the gradient `demb[k, j]` (a 10-dimensional vector) must be routed back to the row `C[Xb[k,j]]` that produced it. We start with `dC = zeros_like(C)` (shape `(27, 10)`) and then loop over all `(k, j)` positions, accumulating with `+=`: `dC[Xb[k,j]] += demb[k,j]`. The `+=` is critical because multiple positions may reference the same row of `C` (e.g., the letter 'a' might appear many times in the batch), and all their gradients must be summed into that one row.

```python
# dembcat: (32, 30) — matmul backward: dembcat = dhprebn @ W1.T
dembcat = dhprebn @ W1.T
# dW1: (30, 64) — matmul backward: dW1 = embcat.T @ dhprebn
dW1     = embcat.T @ dhprebn
# db1: (64,) — sum over batch (undo broadcast-add)
db1     = dhprebn.sum(0)

# demb: (32, 3, 10) — just reshape dembcat back to emb's original shape
demb = dembcat.view(emb.shape)

# dC: (27, 10) — scatter gradients back to the embedding table rows
# += because the same character (row) may be used in many positions
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

**`dlogits` (fused)** — Forward: `loss = F.cross_entropy(logits, Yb)`, which internally computes `softmax(logits)`, takes `log`, selects the correct label, negates, and averages. Instead of back-propagating through each sub-step, we use the analytical result:

- Start with `F.softmax(logits, 1)`, shape `(32, 27)`. This gives us the predicted probability for each character in each example.
- At the correct label position `[e, Yb[e]]`, subtract 1. This is because the derivative of `-log(p_correct)` w.r.t. the logit of the correct class is `p_correct - 1` (i.e., softmax minus 1).
- Divide everything by `n` because the loss is a **mean** over the batch.
- The result is `(32, 27)`. For wrong classes the gradient is a small positive number (push probability down); for the correct class it's a negative number (push probability up).

The grayscale image shows `dlogits` as a grid: the correct label positions appear as dark (negative) squares against a lighter (near-zero positive) background.

```python
# fused cross-entropy: one forward call replaces the entire loss sub-graph
loss_fast = F.cross_entropy(logits, Yb)

# dlogits: (32, 27) — softmax gives predicted probs; subtract 1 at correct label; divide by n
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

**`dhprebn` (fused)** — Instead of stepping through `dbndiff`, `dbnvar_inv`, `dbnvar`, `dbndiff2`, `dbnmeani` individually, we combine them into one formula. Given `dhpreact` (shape `(32, 64)`), `bngain` (shape `(1, 64)`), `bnvar_inv` (shape `(1, 64)`), and `bnraw` (shape `(32, 64)`):

The formula has three terms inside the parentheses:
1. **`n * dhpreact`** — the direct gradient path, scaled up by `n` because we'll divide by `n` outside. Shape `(32, 64)`.
2. **`- dhpreact.sum(0)`** — the mean-subtraction correction. `dhpreact.sum(0)` collapses `(32, 64)` → `(64,)`, then broadcasts back to `(32, 64)`. This subtracts the average gradient, reflecting how batch-norm centres each neuron.
3. **`- n/(n-1) * bnraw * (dhpreact * bnraw).sum(0)`** — the variance correction. `(dhpreact * bnraw).sum(0)` is the correlation of gradients with normalised values, shape `(64,)`. Multiplying by `bnraw` (shape `(32, 64)`) and broadcasting. The `n/(n-1)` factor comes from Bessel's correction in the variance.

The outer `bngain * bnvar_inv / n` multiplies the whole expression: `bngain` and `bnvar_inv` are both `(1, 64)`, the product is `(1, 64)`, and it broadcasts to the `(32, 64)` result. Dividing by `n` normalises for the batch size.

```python
# dhprebn (fused): (32, 64) — entire batch-norm backward in one expression
# Three terms: direct path, mean correction, variance correction
dhprebn = bngain * bnvar_inv / n * (
    n * dhpreact                                          # direct gradient scaled by n
    - dhpreact.sum(0)                                     # mean correction: (64,) broadcasts to (32,64)
    - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0)         # variance correction with Bessel's factor
)

cmp('hprebn', dhprebn, hprebn)   # approximate=True, maxdiff ≈ 9e-10
```

---

### Part 4: Full Manual Training Loop

**Exercise 10**

Wrapping the entire loop in `with torch.no_grad()` tells PyTorch that no computational graph needs to be built. This is correct because we are computing all gradients ourselves; we are only using PyTorch as an efficient tensor engine. The manual gradients are arranged in the same order as `parameters`, so they can be zipped and applied directly without accessing `.grad`.

The backward pass inside the training loop chains together the fused versions from Exercises 8 and 9, plus the atomic steps from Exercises 4, 5, and 7. Here is what each line does:

- **`dlogits`**: Fused cross-entropy backward (Exercise 8). `F.softmax(logits, 1)` gives `(32, 27)`, subtract 1 at correct labels, divide by `n`.
- **`dh`**: `dlogits @ W2.T` — pull gradient back through second linear layer. `(32, 27) @ (27, 200)` → `(32, 200)`.
- **`dW2`**: `h.T @ dlogits` — gradient of `W2`. `(200, 32) @ (32, 27)` → `(200, 27)`.
- **`db2`**: `dlogits.sum(0)` — gradient of `b2`, sum over batch. `(32, 27)` → `(27,)`.
- **`dhpreact`**: `(1 - h**2) * dh` — tanh backward. `(32, 200)` element-wise.
- **`dbngain`**: `(bnraw * dhpreact).sum(0, keepdim=True)` — batch-norm gain gradient. `(32, 200)` → `(1, 200)`.
- **`dbnbias`**: `dhpreact.sum(0, keepdim=True)` — batch-norm bias gradient. `(32, 200)` → `(1, 200)`.
- **`dhprebn`**: Fused batch-norm backward (Exercise 9). Three-term formula producing `(32, 200)`.
- **`dembcat`**: `dhprebn @ W1.T` — pull gradient back through first linear layer. `(32, 200) @ (200, 30)` → `(32, 30)`.
- **`dW1`**: `embcat.T @ dhprebn` — gradient of `W1`. `(30, 32) @ (32, 200)` → `(30, 200)`.
- **`db1`**: `dhprebn.sum(0)` — gradient of `b1`. `(32, 200)` → `(200,)`.
- **`demb`**: Reshape `dembcat` from `(32, 30)` → `(32, 3, 10)`.
- **`dC`**: Scatter `demb` back into the embedding table `(27, 10)` using `+=` in a double loop.

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
