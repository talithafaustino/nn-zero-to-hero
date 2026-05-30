## Questions

This guide follows the chronological flow of the video tutorial for building **makemore**, a character-level language model. Each exercise is designed for you to implement the code yourself based on the logic described in the sources. You can find the complete implementation for reference in the [makemore_part1_bigrams.ipynb GitHub repository](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb).

### **Part 1: Data Exploration and Preprocessing**

**Exercise 1: Loading the Dataset**  
Open the `names.txt` file, read all the names into a single string, and then split them into a Python list of strings where each element is a name.  
*   **Goal:** Verify you have approximately 32,000 names.
*   **Check:** Find and print the length of the shortest and longest names in the dataset.

**Exercise 2: Identifying Bigrams**  
Iterate through the first few names and identify every "bigram" (a pair of consecutive characters).  
*   **Requirement:** Use the `zip` function to create tuples of `(character1, character2)`.
*   **Special Tokens:** To help the model understand the start and end of names, wrap each word with a special start and end token (e.g., use `<S>` and `<E>` or a simple `.`).

---

### **Part 2: The Counting-Based Model**

**Exercise 3: Building a Count Dictionary**  
Create a Python dictionary where the keys are bigram tuples and the values are the number of times that bigram appears across the entire dataset.
*   **Logic:** For every word, add start/end markers and increment the count for every consecutive pair.
*   **Exploration:** Sort the dictionary by frequency to see which character pairs (like `n` at the end of a name) are most common.

**Exercise 4: Moving to Tensors**  
Instead of a dictionary, store these counts in a 2D **PyTorch tensor**.  
*   **Mapping:** Create a lookup table (dictionary) that maps every character (a–z plus your special token) to an integer index (0–26) and vice versa.
*   **Implementation:** Initialize a 27x27 tensor of zeros (using `torch.int32`) and populate it by iterating through all words and incrementing the corresponding row/column index.

**Exercise 5: Visualizing the Counts**  
Use `matplotlib` to visualize the 27x27 grid.  
*   **Task:** Overlay the bigram text and the raw count on each cell of the grid to see the statistical structure of the names.

---

### **Part 3: Sampling and Evaluation**

**Exercise 6: Generating Names by Sampling**  
Implement a loop to generate new names using the counts in your tensor.  
*   **Step A:** Convert the raw counts in a row into a probability distribution by dividing each element by the sum of the row.
*   **Step B:** Use `torch.multinomial` to sample the next character based on these probabilities.
*   **Step C:** Repeat this process, starting from the "start" token, until you sample the "end" token.

**Exercise 7: Efficient Matrix Operations (Broadcasting)**  
To avoid recalculating probabilities in every loop, create a dedicated probability matrix `P` where every row sums to 1.  
*   **Constraint:** Use `P = (N+1).float()` (adding 1 for **model smoothing** to avoid zero-probabilities) and divide by the row sums.
*   **Implementation Detail:** Use `torch.sum(1, keepdim=True)` and ensure you understand how **broadcasting** allows a 27x27 matrix to be divided by a 27x1 vector.

**Exercise 8: Calculating the Loss (Negative Log Likelihood)**  
Evaluate the quality of your model using a single numerical value.  
*   **Concept:** For every bigram in your dataset, find the probability the model assigned to it.
*   **Calculation:** Calculate the **log likelihood** by summing the logs of these probabilities.
*   **Result:** The loss is the **negative average log likelihood**. A lower value means a better model.

---

### **Part 4: The Neural Network Approach**

**Exercise 9: Preparing the Training Set**  
Re-cast the bigram problem into a supervised learning task.
*   **Inputs (X):** A tensor of the first character's integer index for every bigram.
*   **Labels (Y):** A tensor of the second character's integer index for every bigram.
*   **Check:** If training on just the name "emma", verify you have 5 input-label pairs.

**Exercise 10: One-Hot Encoding**  
Because you cannot feed raw integers into a neural network, convert your input tensor `X` into a **one-hot encoded** tensor.
*   **Implementation:** Use `torch.nn.functional.one_hot` and cast the result to `float32`.

**Exercise 11: The Forward Pass (Linear Layer & Softmax)**  
Build the simplest possible neural network: a single linear layer.
*   **Weights:** Initialize a 27x27 weight matrix `W` with random values.
*   **Matrix Multiplication:** Multiply your one-hot inputs by `W` to get "logits".
*   **Softmax:** Exponentiate the logits and normalize them to produce a probability distribution for every input example.

**Exercise 12: Optimization (The Training Loop)**  
Implement the gradient descent loop to tune the weights `W`.
*   **Zero Grad:** Set gradients to `None`.
*   **Backward Pass:** Call `.backward()` on your negative log likelihood loss to compute the gradients of the weights.
*   **Update:** Nudge the weights in the opposite direction of the gradient using a learning rate (e.g., `0.1` or `50`).
*   **Goal:** Watch the loss decrease over several iterations until it matches the loss from the counting-based model.

---

### **Part 5: Advanced Challenges (Trigram Exercises)**

Once you have implemented the bigram model, try these extensions based on the additional sources:

1.  **Trigram Model:** Train a model that takes *two* characters as input to predict the 3rd one. Evaluate if the loss improves over the bigram version.
2.  **Dataset Splitting:** Split your data into 80% training, 10% development, and 10% test sets. Train only on the training set and evaluate on the others.
3.  **Smoothing Tuning:** Use the development set to find the best smoothing/regularization strength for your trigram model.
4.  **Optimization Trick:** Remove the use of `F.one_hot` and instead use the integer indices to directly index into the rows of `W`.
5.  **Efficient Loss:** Replace your manual negative log likelihood calculation with `F.cross_entropy`.

## Answers

This document provides the implementation answers for the exercises derived from the **makemore** tutorial. For the complete original code, refer to the [makemore_part1_bigrams.ipynb GitHub repository](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb).

### **Part 1: Data Exploration and Preprocessing Answers**

**Exercise 1: Loading the Dataset**
```python
words = open('names.txt', 'r').read().splitlines() #
print(len(words)) # Should be ~32,033
print(min(len(w) for w in words)) # 2
print(max(len(w) for w in words)) # 15
```

**Exercise 2: Identifying Bigrams**
```python
for w in words[:3]:
    chs = ['.'] + list(w) + ['.'] # Using '.' as both start and end token
    for ch1, ch2 in zip(chs, chs[1:]): #
        print(ch1, ch2) #
```

---

### **Part 2: The Counting-Based Model Answers**

**Exercise 3: Building a Count Dictionary**
```python
b = {}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1 #

# Sort to see common bigrams
sorted(b.items(), key = lambda kv: -kv) #
```

**Exercise 4: Moving to Tensors**
```python
import torch #

# Mapping setup
chars = sorted(list(set(''.join(words))))
s2i = {s:i+1 for i,s in enumerate(chars)}
s2i['.'] = 0
i2s = {i:s for s,i in s2i.items()} #

# Populate 27x27 tensor
N = torch.zeros((27, 27), dtype=torch.int32) #
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = s2i[ch1], s2i[ch2] #
        N[ix1, ix2] += 1 #
```

**Exercise 5: Visualizing the Counts**
```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = i2s[i] + i2s[j] #
        plt.text(j, i, chstr, va='bottom', ha='center', color='gray') #
        plt.text(j, i, N[i, j].item(), va='top', ha='center', color='gray') #
plt.axis('off');
```

---

### **Part 3: Sampling and Evaluation Answers**

**Exercise 6: Generating Names by Sampling**
```python
g = torch.Generator().manual_seed(2147483647) # For deterministic results

for i in range(5):
    out = []
    ix = 0 # Start with '.'
    while True:
        p = N[ix].float() #
        p = p / p.sum() # Normalize to probabilities
        
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() #
        out.append(i2s[ix])
        if ix == 0: # End token reached
            break
    print(''.join(out)) #
```

**Exercise 7: Efficient Matrix Operations (Broadcasting)**
```python
# Smoothing: add 1 to counts to avoid zero probabilities (and infinite loss)
P = (N+1).float() 
# Normalize rows: Use keepdim=True to ensure correct broadcasting (27,27) / (27,1)
P /= P.sum(1, keepdim=True) #
```

**Exercise 8: Calculating the Loss (Negative Log Likelihood)**
```python
log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1, ix2 = s2i[ch1], s2i[ch2]
        prob = P[ix1, ix2] #
        logprob = torch.log(prob) #
        log_likelihood += logprob #
        n += 1

nll = -log_likelihood #
print(f'{nll/n:.4f}') # Average NLL (Loss)
```

---

### **Part 4: The Neural Network Approach Answers**

**Exercise 9: Preparing the Training Set**
```python
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(s2i[ch1])
        ys.append(s2i[ch2])
        
xs = torch.tensor(xs) #
ys = torch.tensor(ys) #
num = xs.nelement() #
```

**Exercise 10: One-Hot Encoding**
```python
import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float() #
```

**Exercise 11: The Forward Pass (Linear Layer & Softmax)**
```python
W = torch.randn((27, 27), requires_grad=True) #

# Logic: matrix mult -> exponentiate (counts) -> normalize (probs)
logits = xenc @ W #
counts = logits.exp() #
probs = counts / counts.sum(1, keepdims=True) # Softmax
```

**Exercise 12: Optimization (The Training Loop)**
```python
for k in range(100): # Run for 100 iterations
    # Forward Pass
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    
    # Loss: Negative Log Likelihood
    loss = -probs[torch.arange(num), ys].log().mean() #
    
    # Backward Pass
    W.grad = None #
    loss.backward() #
    
    # Update
    W.data += -50 * W.grad # Learning rate of 50
```

---

### **Part 5: Advanced Trigram Challenges Answers**

Based on the proposed exercises in the sources:

*   **E01 (Trigram Model):** To implement this, your input `X` becomes a pair of characters. Your count matrix `N` would be `(27*27) x 27`. The loss should be lower than the bigram model as it has more context.
*   **E02 (Data Splits):** Use indexing to split: `n1 = int(0.8*len(words))`, `n2 = int(0.9*len(words))`. Train on `words[:n1]`, validate on `words[n1:n2]`, and test on `words[n2:]`.
*   **E03 (Smoothing Tuning):** In the counting model, vary the `+1` (e.g., `+0.1`, `+0.01`). In the neural net, add a regularization term: `loss += 0.01 * (W**2).mean()`.
*   **E04 (Index vs One-Hot):** Replace `xenc @ W` with `logits = W[xs]`. Pytorch's indexing is equivalent to multiplying a one-hot vector by a weight matrix.
*   **E05 (Cross Entropy):** Use `F.cross_entropy(logits, ys)`. This is more numerically stable than manual softmax because it handles log-sum-exp internally.