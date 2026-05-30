## Questions

These exercises follow the chronological flow of the **Let's build the GPT Tokenizer** video and the notebook [nanogpt/tokenization.ipynb](nanogpt/tokenization.ipynb). Work through them in order to build a complete byte-pair encoding tokenizer from scratch.

---

### Part 1: Unicode and UTF-8 Foundations

**Exercise 1: Exploring Unicode Code Points**  
Write code to explore how Python represents text internally:
- Create a string that mixes English, Korean (`"안녕하세요"`), and an emoji (`"👋"`), e.g. `"안녕하세요 👋 (hello in Korean!)"`.
- Use `ord()` on each character to obtain the Unicode code point as an integer. Print the resulting list of integers.
- Explain what a Unicode code point is and roughly how many characters the Unicode standard currently defines.

**Exercise 2: UTF-8 Encoding**  
Convert the same mixed-language string into its raw byte representation:
- Call `.encode("utf-8")` on the string, then convert the result into a list of integers (each in the range 0–255).
- Print the byte list and its length.
- Compare the length of the code-point list (from Exercise 1) with the byte list. Why is the byte list typically longer?
- Briefly explain why UTF-8 is preferred over UTF-16 or UTF-32 for our purposes (backward compatibility with ASCII, no wasted zero bytes, variable-length efficiency).

**Exercise 3: Preparing a Longer Training Text**  
Prepare a longer text for BPE training:
- Take a multi-paragraph block of text (e.g. the Unicode blog post excerpt used in the reference notebook, or any long English passage of your choice).
- Encode it to UTF-8 and convert to a list of integers.
- Print the original text length (in code points) and the encoded byte length. Confirm that the byte length is ≥ the code-point length.

---

### Part 2: Byte Pair Encoding — Core Algorithm

**Exercise 4: Counting Byte Pair Frequencies**  
Write a `get_stats(ids)` function that:
- Takes a list of integers (`ids`).
- Iterates over all consecutive pairs using `zip(ids, ids[1:])`.
- Returns a dictionary mapping each pair (tuple of two ints) to the number of times it occurs.
- Call `get_stats` on your encoded training tokens and identify the most frequent pair. Print the pair and its count. Use `chr()` to display what characters that pair represents.

**Exercise 5: Merging a Single Pair**  
Write a `merge(ids, pair, idx)` function that:
- Takes a list of integers `ids`, a `pair` tuple `(a, b)`, and a new token integer `idx`.
- Returns a new list where every consecutive occurrence of `pair` is replaced by `idx`.
- Handle the boundary condition: do not go out of bounds at the last position.
- Test on a small example: `merge([5, 6, 6, 7, 9, 1], (6, 7), 99)` should return `[5, 6, 99, 9, 1]`.
- Apply it once to the training tokens: merge the top pair into token `256`. Verify that the length decreased by the pair's count, and that the original pair no longer appears in the new list.

**Exercise 6: Iterative BPE Training Loop**  
Write the full BPE training loop:
- Set a desired `vocab_size` (e.g. `276`, meaning `20` merges on top of `256` raw byte tokens).
- Compute `num_merges = vocab_size - 256`.
- Copy the tokens list. Create an empty `merges` dictionary.
- In a loop of `num_merges` iterations:
  - Find the most frequent pair using `get_stats`.
  - Assign it the next available token ID (`256 + i`).
  - Merge that pair in the working token list.
  - Record the merge in the `merges` dictionary: `merges[pair] = idx`.
  - Print the merge being performed.
- After the loop, print the compression ratio: `len(original_tokens) / len(compressed_tokens)`.

---

### Part 3: Decoding — Tokens to Text

**Exercise 7: Building the Vocabulary Lookup**  
Build a `vocab` dictionary that maps each token ID to its byte representation:
- Start with the 256 raw byte tokens: `vocab = {idx: bytes([idx]) for idx in range(256)}`.
- Iterate through the `merges` dictionary **in insertion order** (guaranteed since Python 3.7). For each `(p0, p1) → idx`, set `vocab[idx] = vocab[p0] + vocab[p1]` (bytes concatenation).
- Print a few entries from `vocab` (e.g. token `256`) to verify.

**Exercise 8: Implementing `decode`**  
Write a `decode(ids)` function that:
- Takes a list of integer token IDs.
- Looks up each ID in `vocab` and concatenates the resulting bytes.
- Calls `.decode("utf-8", errors="replace")` on the concatenated bytes to produce a Python string.
- Test: `decode([128])` should return the Unicode replacement character `�`, not raise an error. Explain why not every byte sequence is valid UTF-8 and why `errors="replace"` is the standard practice.

---

### Part 4: Encoding — Text to Tokens

**Exercise 9: Implementing `encode`**  
Write an `encode(text)` function that:
- Encodes the input string into UTF-8 bytes and converts to a list of integers.
- In a while loop (while there are at least 2 tokens):
  - Calls `get_stats` on the current tokens.
  - Finds the pair with the **lowest** index in `merges` using `min(stats, key=lambda p: merges.get(p, float("inf")))`.
  - If the best pair is not in `merges`, breaks (nothing more to merge).
  - Otherwise, retrieves `idx = merges[pair]` and calls `merge(tokens, pair, idx)`.
- Returns the final list of tokens.
- Handle the edge case: if the input text is empty or a single character (fewer than 2 tokens), return immediately.

**Exercise 10: Round-Trip Verification**  
Verify that encoding and decoding are inverses:
- For the training text, check `decode(encode(text)) == text`. It should be `True`.
- For a separate validation text the tokenizer has never seen, check the same round-trip property.
- Explain why `encode(decode(ids)) == ids` is **not** guaranteed in general (hint: not all token sequences correspond to valid UTF-8).

---

### Part 5: Regex-Based Splitting (GPT-2 Tokenizer)

**Exercise 11: Understanding the GPT-2 Regex Pattern**  
Study the regex pattern used by GPT-2 to split text before applying BPE:
```python
import regex as re
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```
- Apply `re.findall(gpt2pat, "Hello've world123 how's   are you!!!?")` and inspect the resulting list.
- Explain what each alternative in the pattern matches:
  - `'s|'t|'re|'ve|'m|'ll|'d` — common English contractions.
  - `?\p{L}+` — an optional space followed by one or more Unicode letters.
  - `?\p{N}+` — an optional space followed by one or more Unicode digits.
  - `?[^\s\p{L}\p{N}]+` — an optional space followed by one or more punctuation characters.
  - `\s+(?!\S)` — whitespace up to but not including the last whitespace character (negative lookahead).
  - `\s+` — any remaining whitespace.
- Explain why this splitting prevents merges across category boundaries (e.g., letters will never merge with punctuation, preventing tokens like `dog.`).

**Exercise 12: Splitting a Python Code Snippet**  
Apply the GPT-2 regex pattern to a Python code snippet (e.g. FizzBuzz):
```python
example = """
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
"""
print(re.findall(gpt2pat, example))
```
- Observe how each individual space is a separate element.
- Explain why this is problematic for GPT-2's handling of Python: every indentation space becomes its own token, wasting context length.

---

### Part 6: GPT-4 Tokenizer Improvements

**Exercise 13: Comparing GPT-2 and GPT-4 Tokenization with `tiktoken`**  
Use the `tiktoken` library to compare:
```python
import tiktoken

enc_gpt2 = tiktoken.get_encoding("gpt2")
enc_gpt4 = tiktoken.get_encoding("cl100k_base")

test_str = "    hello world!!!"
print("GPT-2:", enc_gpt2.encode(test_str))
print("GPT-4:", enc_gpt4.encode(test_str))
```
- Note the token count difference. GPT-4 uses roughly 100K vocabulary vs GPT-2's ~50K.
- Explain the key improvements in GPT-4's tokenizer:
  - Case-insensitive apostrophe matching (the `(?i)` flag).
  - Merging of whitespace sequences (spaces are grouped, not kept individual).
  - Numbers are matched in groups of 1–3 digits only, preventing very long numeric tokens.
- Why does doubling the vocabulary size roughly halve the number of tokens for the same text?

**Exercise 14: Inspecting the GPT-2 Saved Tokenizer**  
Download and inspect the GPT-2 tokenizer files:
- Load `encoder.json` (the vocabulary mapping: string → token ID, equivalent to our `vocab` inverted).
- Load `vocab.bpe` (the merge list, equivalent to our `merges`).
- Verify `len(encoder)` is `50257` and explain the composition: 256 raw byte tokens + 50,000 merges + 1 special token.
- Identify the single special token in the encoder and print its ID.

---

### Part 7: Special Tokens

**Exercise 15: Understanding Special Tokens**  
Explain the role of special tokens in LLM tokenization:
- What is the `<|endoftext|>` token used for in GPT-2? Why is it needed for training on concatenated documents?
- How do special tokens differ from BPE-derived tokens? (They are not produced by the merge algorithm; they are matched by special-case logic before BPE runs.)
- In GPT-4's `cl100k_base` tokenizer, list the additional special tokens beyond `<|endoftext|>` (FIM prefix, middle, suffix, and one more). What does FIM stand for?

**Exercise 16: Extending a Tokenizer with Custom Special Tokens**  
Using `tiktoken`'s extension API (as documented in the tiktoken README), show conceptually how you would add a new special token (e.g. `<|custom|>`) to a tokenizer:
- Explain the model surgery required when adding new tokens: the embedding matrix gains a new row (initialized with small random numbers), and the final linear projection layer gains a new output dimension.
- Why is it common to freeze the base model weights and only train the new token embeddings during fine-tuning?

---

### Part 8: SentencePiece Tokenizer

**Exercise 17: Training a SentencePiece BPE Tokenizer**  
Use the `sentencepiece` library to train a small BPE tokenizer:
- Write a toy training text to a file `toy.txt`.
- Train with settings mirroring Llama 2:
  - `model_type="bpe"`, `vocab_size=400`.
  - Turn off normalization: `normalization_rule_name="identity"`, `remove_extra_whitespaces=False`.
  - Enable `byte_fallback=True`, `split_digits=True`.
  - Set special tokens: `unk_id=0`, `bos_id=1`, `eos_id=2`, `pad_id=-1`.
- Load the trained model and print the full vocabulary.
- Describe the vocabulary ordering: special tokens → byte fallback tokens (256) → merge tokens → individual code-point tokens.

**Exercise 18: SentencePiece vs tiktoken — Key Differences**  
Encode the string `"hello 안녕하세요"` with your trained SentencePiece model:
- Print the token IDs and the decoded pieces.
- Observe how Korean characters that were not in the training data are handled:
  - With `byte_fallback=True`: they are encoded as raw UTF-8 bytes using the byte tokens.
  - With `byte_fallback=False`: they become the `<unk>` token (ID 0).
- Explain the fundamental difference in approach:
  - **tiktoken**: encodes text to UTF-8 bytes first, then runs BPE on bytes.
  - **SentencePiece**: runs BPE directly on Unicode code points, falling back to byte tokens for rare code points.
- Explain what the `add_dummy_prefix` option does and why Llama 2 uses it (to make `"world"` and `" world"` produce the same tokenization).

---

### Part 9: Vocab Size Considerations and LLM Impact

**Exercise 19: Where Vocab Size Appears in the Transformer**  
Referring to a Transformer architecture (e.g. the GPT model from the previous lecture):
- Identify the **two** places where `vocab_size` affects model parameters:
  1. The token embedding table (`nn.Embedding(vocab_size, n_embd)`).
  2. The final linear projection / LM head (`nn.Linear(n_embd, vocab_size)`).
- Explain the trade-offs of increasing vocab size:
  - **Pro**: shorter sequences → more context visible in the attention window.
  - **Con**: larger embedding table and LM head → more parameters and compute; each token is seen less often → under-training risk; information gets squished into single tokens → less "thinking time" per character.

**Exercise 20: Extending the Vocabulary of a Pre-Trained Model**  
Describe the procedure for adding new tokens to a pre-trained Transformer:
- Resize the embedding matrix by adding new rows initialized to small random numbers.
- Resize the final projection layer to produce logits for the new token(s).
- Optionally freeze all existing parameters and train only the new embeddings.
- Mention the "gist tokens" idea (from the paper *Learning to Compress Prompts*): new tokens whose embeddings are optimized by distillation to stand in for long prompts.

---

### Part 10: Tokenization Pitfalls and Practical Advice

**Exercise 21: Why LLMs Struggle with Spelling and String Reversal**  
Using GPT-4's tokenizer, demonstrate why LLMs have trouble with character-level tasks:
- Find a long token in the GPT-4 vocabulary (e.g. `" DefaultCellStyle"` is a single token).
- Explain why the model cannot easily count or reverse the letters within such a token — it sees the whole chunk as one atomic unit.
- Describe the workaround: ask the model to first list out each character separated by spaces, then perform the reversal on that list.

**Exercise 22: Non-English Tokenization Blow-Up**  
Compare the tokenization of an English sentence and its Korean translation:
- Show that `"hello how are you"` tokenizes to roughly 5 tokens, while its Korean equivalent may use ~15 tokens.
- Explain the consequences: non-English text is stretched out in the token sequence, consuming more of the finite context window, and the model sees fewer examples per token during training.

**Exercise 23: The Trailing-Whitespace Problem**  
Explain the "trailing whitespace" warning from the OpenAI API:
- In GPT-style tokenizers, a leading space is typically part of the next word's token (e.g. `" our"` is a single token).
- If you end your prompt with a trailing space, you create a rare token (bare space, token 220 in GPT-2) that the model has almost never seen in isolation during training.
- This puts the model out-of-distribution and can cause degraded completions or even immediate stop-sequence predictions.

**Exercise 24: The SolidGoldMagikarp Phenomenon**  
Explain the "SolidGoldMagikarp" phenomenon:
- A Reddit user `SolidGoldMagikarp` appeared frequently in the tokenizer's training data, so the username was merged into a single token.
- That token was **never seen** in the LLM's training data (different dataset), so its embedding row was never updated from its random initialization.
- At inference time, invoking this token feeds "unallocated memory" into the Transformer, producing undefined behavior: hallucinations, evasion, insults, or safety violations.
- What is the root cause? A mismatch between the tokenizer's training corpus and the language model's training corpus.

**Exercise 25: JSON vs YAML Token Efficiency**  
Compare the token counts of equivalent data in JSON and YAML:
- Encode the same structured data in both formats and tokenize with the GPT-4 tokenizer.
- Show that YAML is typically more token-efficient (fewer tokens for the same information).
- Why does this matter in the "token economy" — both in terms of API cost per token and in terms of effective context length?

---

## Answers

---

### Part 1: Unicode and UTF-8 Foundations

**Answer 1: Exploring Unicode Code Points**

Unicode is a standard maintained by the Unicode Consortium that defines roughly 150,000 characters across 161 scripts. Each character is assigned a unique integer called a **code point**. In Python, strings are sequences of Unicode code points, and you can retrieve any character's code point with `ord()`.

As Karpathy explains: "What are Unicode code points? Unicode code points are defined by the Unicode Consortium as part of the Unicode standard. It's just a definition of roughly 150,000 characters right now and roughly speaking what they look like and what integers represent those characters."

```python
s = "안녕하세요 👋 (hello in Korean!)"
print(s)
print([ord(x) for x in s])
```

The output will show small integers for ASCII characters (e.g. 104 for `'h'`), larger integers for Korean characters (e.g. around 50,000), and very large integers for emoji (e.g. around 128,000).

**Answer 2: UTF-8 Encoding**

UTF-8 is a variable-length encoding that converts each Unicode code point into 1–4 bytes. Simple ASCII characters map to a single byte, while more complex characters (Korean, emoji) require 2–4 bytes. This is why the byte list is typically longer than the code-point list.

Karpathy explains the preference for UTF-8: "UTF-8 is the only one of these that is backwards compatible to the much simpler ASCII encoding of text." UTF-16 wastes space on English text (lots of zero bytes), and UTF-32 uses 4 bytes per character unconditionally. UTF-8 is the most efficient for typical mixed text.

```python
s = "안녕하세요 👋 (hello in Korean!)"
tokens = list(s.encode("utf-8"))
print(tokens)
print("length:", len(tokens))
```

**Answer 3: Preparing a Longer Training Text**

We take a longer text so that pair statistics are more representative. As Karpathy notes: "making the training text longer will allow us to have more representative statistics for the byte pairs and we'll just get more sensible results out of it because it's longer text."

```python
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺\u200c🇳\u200c🇮\u200c🇨\u200c🇴\u200c🇩\u200c🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to \"support Unicode\" in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don't blame programmers for still finding the whole thing mysterious, even 30 years after Unicode's inception."
tokens = text.encode("utf-8")
tokens = list(map(int, tokens))
print('---')
print(text)
print("text length (code points):", len(text))
print('---')
print(tokens)
print("byte length:", len(tokens))
```

---

### Part 2: Byte Pair Encoding — Core Algorithm

**Answer 4: Counting Byte Pair Frequencies**

The first step of BPE is to find which consecutive byte pairs occur most frequently in the token stream. We then merge the most common pair into a new token, iteratively compressing the sequence.

As Karpathy explains: "We iteratively find the pair of tokens that occur the most frequently, and then once we've identified that pair we replace that pair with just a single new token that we append to our vocabulary."

```python
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
top_pair = max(stats, key=stats.get)
print(f"Most common pair: {top_pair}, count: {stats[top_pair]}")
print(f"Characters: {chr(top_pair[0])!r} + {chr(top_pair[1])!r}")
```

**Answer 5: Merging a Single Pair**

The merge function scans the list from left to right. When it finds the target pair, it replaces both elements with the new index and skips ahead by 2. The boundary check `i < len(ids) - 1` prevents an out-of-bounds access at the last position.

```python
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# Small test
print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))  # [5, 6, 99, 9, 1]

# Apply to training tokens
tokens2 = merge(tokens, top_pair, 256)
print("original length:", len(tokens))
print("new length:", len(tokens2))
print("reduction:", len(tokens) - len(tokens2))
```

**Answer 6: Iterative BPE Training Loop**

We repeat the find-and-merge cycle for a desired number of merges. The `merges` dictionary records all merges and is the core artifact of a trained tokenizer. As Karpathy describes: "the parameters of this tokenizer really are just this dictionary of merges, and that basically creates the little binary forest on top of raw bytes."

Importantly, newly minted tokens are eligible for further merging: "every time we replace these tokens they become eligible for merging in the next round of iteration, so that's why we're building up a small sort of binary forest instead of a single individual tree."

```python
vocab_size = 276  # desired final vocabulary size
num_merges = vocab_size - 256
ids = list(tokens)  # copy so we don't destroy the original list

merges = {}  # (int, int) -> int
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx

print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")
```

---

### Part 3: Decoding — Tokens to Text

**Answer 7: Building the Vocabulary Lookup**

We build the vocab bottom-up: the first 256 entries are single bytes, then each merge concatenates the bytes of its two children. Iteration order matters — Python 3.7+ guarantees dictionaries maintain insertion order.

As Karpathy notes: "it really matters that this runs in the order in which we inserted items into the merges dictionary. Luckily starting with Python 3.7 this is guaranteed to be the case."

```python
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

print(vocab[256])  # bytes representation of first merge
```

**Answer 8: Implementing `decode`**

Not every byte sequence is valid UTF-8. For example, the byte `128` (binary `10000000`) is an invalid start byte under UTF-8's encoding rules. If the LLM predicts tokens in an unfortunate order, the resulting bytes may not decode cleanly. The `errors="replace"` option substitutes the Unicode replacement character `�` instead of raising an error.

Karpathy explains: "not every single byte sequence is valid UTF-8, and if it happens that your large language model predicts tokens in a bad manner then they might not fall into valid UTF-8 and then we won't be able to decode them."

```python
def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

print(decode([128]))  # prints the replacement character
```

---

### Part 4: Encoding — Text to Tokens

**Answer 9: Implementing `encode`**

The encoding function converts text to bytes, then greedily applies merges in priority order (lowest merge index first). Using `min` with `merges.get(p, float("inf"))` ensures that:
1. Only pairs that exist in `merges` are considered (others get `inf`).
2. Among eligible pairs, the one with the lowest merge index (earliest merge) is applied first.

This ordering is critical because later merges may depend on tokens created by earlier merges. As Karpathy explains: "we prefer to do all these merges in the beginning before we do these merges later because this merge over here relies on the 256 which got merged here, so we have to go in the order from top to bottom."

```python
def encode(text):
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens

print(encode("hello world"))
```

**Answer 10: Round-Trip Verification**

Encoding then decoding should always recover the original string, because `encode` faithfully translates text → bytes → merged tokens and `decode` reverses this exactly. However, the reverse direction (decode then encode) is not guaranteed because arbitrary token sequences might not correspond to valid UTF-8, and `errors="replace"` introduces replacement characters that were not in the original.

```python
# Training text round-trip
print(decode(encode(text)) == text)  # True

# Validation text round-trip
valtext = "Many common characters, including numerals, punctuation, and other symbols, are unified within the standard and are not treated as specific to any given writing system."
print(decode(encode(valtext)) == valtext)  # True
```

---

### Part 5: Regex-Based Splitting (GPT-2 Tokenizer)

**Answer 11: Understanding the GPT-2 Regex Pattern**

GPT-2 uses a regex pattern to split text into chunks before running BPE. Each chunk is processed independently, which prevents merges from happening across category boundaries. Karpathy explains: "using this regex pattern to chunk up the text is just one way of enforcing that some merges are not to happen. What this is trying to do on a high level is we're trying to not merge across letters, across numbers, across punctuation, and so on."

The motivation comes from the GPT-2 paper: without forced splits, common words like `"dog"` could merge with adjacent punctuation to create tokens like `"dog."`, `"dog!"`, `"dog?"` — wasting vocabulary capacity on semantically meaningless combinations.

```python
import regex as re
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
print(re.findall(gpt2pat, "Hello've world123 how's   are you!!!?"))
```

Breakdown of each alternative:
- `'s|'t|'re|'ve|'m|'ll|'d` — common English contractions split off.
- ` ?\p{L}+` — optional space + one or more Unicode letters (matches words).
- ` ?\p{N}+` — optional space + one or more Unicode digits (matches numbers).
- ` ?[^\s\p{L}\p{N}]+` — optional space + one or more punctuation characters.
- `\s+(?!\S)` — whitespace up to but NOT including the last whitespace character (negative lookahead), ensuring the last space can join the next word token.
- `\s+` — fallback: any remaining whitespace.

**Answer 12: Splitting a Python Code Snippet**

```python
example = """
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
"""
print(re.findall(gpt2pat, example))
```

Each individual space becomes its own element in the split list. Since BPE only merges within chunks, and GPT-2 never merges these individual spaces, every indentation space is its own token (token 220). Karpathy explains: "all these individual spaces are all separate tokens, they are token 220 ... the Transformer needs to handle all these spaces individually, they all feed in one by one into the entire Transformer in the sequence, and so this is being extremely wasteful."

---

### Part 6: GPT-4 Tokenizer Improvements

**Answer 13: Comparing GPT-2 and GPT-4 Tokenization**

GPT-4 roughly doubles the vocabulary (from ~50K to ~100K), meaning the same text is compressed into roughly half as many tokens. Key pattern changes in the GPT-4 regex:

1. **Case-insensitive matching** (`(?i)` flag) — contractions like `'S` and `'s` are handled consistently. As Karpathy notes about the GPT-2 comment: "should have added re.ignorecase so BPE merges can happen for capitalized versions of contractions."

2. **Whitespace merging** — multiple spaces are grouped into single tokens instead of remaining individual. This dramatically improves Python tokenization efficiency.

3. **1–3 digit number matching** — prevents very long numeric tokens. "They only match one to three numbers, so they will never merge numbers that are more than three digits."

```python
import tiktoken

enc_gpt2 = tiktoken.get_encoding("gpt2")
enc_gpt4 = tiktoken.get_encoding("cl100k_base")

test_str = "    hello world!!!"
print("GPT-2:", enc_gpt2.encode(test_str), "count:", len(enc_gpt2.encode(test_str)))
print("GPT-4:", enc_gpt4.encode(test_str), "count:", len(enc_gpt4.encode(test_str)))
```

**Answer 14: Inspecting the GPT-2 Saved Tokenizer**

The GPT-2 tokenizer is fully defined by two files:
- `encoder.json` — maps string representations to token IDs (equivalent to our `vocab` but inverted).
- `vocab.bpe` — lists the merge operations in order (equivalent to our `merges`).

The total vocabulary is 50,257 = 256 (raw bytes) + 50,000 (merges) + 1 (the `<|endoftext|>` special token).

```python
import os, json

with open('encoder.json', 'r') as f:
    encoder = json.load(f)

with open('vocab.bpe', 'r', encoding="utf-8") as f:
    bpe_data = f.read()
bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]

print(len(encoder))  # 50257
print(encoder['<|endoftext|>'])  # 50256
```

---

### Part 7: Special Tokens

**Answer 15: Understanding Special Tokens**

Special tokens are not produced by the BPE merge algorithm — they are matched by special-case logic before BPE runs. In GPT-2, `<|endoftext|>` (ID 50256) is inserted between documents in the training data to signal document boundaries. As Karpathy explains: "we are using this as a signal to the language model that the document has ended and what follows is going to be unrelated to the document previously."

In GPT-4's `cl100k_base` tokenizer, additional special tokens include:
- `<|endoftext|>` — same as GPT-2.
- `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>` — used for Fill-In-the-Middle (FIM), a technique for code infilling.
- One additional end-of-prompt token.

These tokens enable special structure in the token stream, such as delimiting conversation turns in chat models (e.g. `<|im_start|>` and `<|im_end|>` for GPT-3.5-turbo).

**Answer 16: Extending a Tokenizer with Custom Special Tokens**

When adding special tokens to a pre-trained model, you must perform minor model surgery:
1. **Embedding table**: add a new row for each new token, initialized with small random numbers.
2. **LM head / final projection**: add a new output dimension for each new token.

As Karpathy describes: "you have to make sure that your embedding matrix for the vocabulary tokens has to be extended by adding a row, and typically this row would be initialized with small random numbers. In addition you have to go to the final layer of the Transformer and make sure that projection at the very end into the classifier is extended by one as well."

It is common to freeze the base model's existing parameters and train only the new token embeddings, because the base model already has strong representations and only the new tokens need to be fitted.

---

### Part 8: SentencePiece Tokenizer

**Answer 17: Training a SentencePiece BPE Tokenizer**

SentencePiece has many configuration options, much of which is historical baggage. For LLM usage, the key is to turn off normalization (keep raw text untouched) and enable byte fallback for rare code points.

Karpathy's assessment: "there's a lot of historical baggage in SentencePiece, a lot of concepts that I think are slightly confusing and potentially contain footguns, like this concept of a sentence and its maximum length."

The vocabulary is ordered: special tokens (unk, bos, eos) → 256 byte fallback tokens → merge tokens → individual code-point tokens.

```python
import sentencepiece as spm

with open("toy.txt", "w", encoding="utf-8") as f:
    f.write("SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end system that does not depend on language-specific pre/postprocessing.")

import os
options = dict(
    input="toy.txt",
    input_format="text",
    model_prefix="tok400",
    model_type="bpe",
    vocab_size=400,
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    input_sentence_size=200000000,
    max_sentence_length=4192,
    seed_sentencepiece_size=1000000,
    shuffle_input_sentence=True,
    character_coverage=0.99995,
    byte_fallback=True,
    split_digits=True,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    max_sentencepiece_length=16,
    add_dummy_prefix=True,
    allow_whitespace_only_pieces=True,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=-1,
    num_threads=os.cpu_count(),
)
spm.SentencePieceTrainer.train(**options)

sp = spm.SentencePieceProcessor()
sp.load('tok400.model')
vocab = [[sp.id_to_piece(idx), idx] for idx in range(sp.get_piece_size())]
print(vocab)
```

**Answer 18: SentencePiece vs tiktoken — Key Differences**

The fundamental difference: **tiktoken** first encodes text to UTF-8 bytes, then runs BPE on those bytes. **SentencePiece** runs BPE directly on Unicode code points, falling back to UTF-8 byte tokens only for rare code points (controlled by the `character_coverage` hyperparameter).

As Karpathy summarizes: "tiktoken encodes to UTF-8 and then BPEs bytes. SentencePiece BPEs the code points and optionally falls back to UTF-8 bytes for rare code points."

The `add_dummy_prefix` option prepends a space to the input so that words at the beginning of a sentence and words in the middle are tokenized identically. As Karpathy explains: "in the tiktoken world, basically words in the beginning of sentences and words in the middle of sentences actually look completely different. add_dummy_prefix is trying to fight that."

```python
ids = sp.encode("hello 안녕하세요")
print(ids)
print([sp.id_to_piece(idx) for idx in ids])
```

Korean characters not in the training set will be encoded as byte fallback tokens (their UTF-8 bytes, shifted by the special token offset).

---

### Part 9: Vocab Size Considerations and LLM Impact

**Answer 19: Where Vocab Size Appears in the Transformer**

Vocab size affects exactly two places in a Transformer:
1. **Token embedding table** — shape `(vocab_size, n_embd)`. Each token has a learnable vector.
2. **LM head** — a linear layer of shape `(n_embd, vocab_size)` that produces logits for the next-token prediction.

Karpathy explains the trade-offs clearly: "As your vocab size grows you're going to start shrinking your sequences a lot, and that's really nice because that means we're going to be attending to more and more text. But also you might be worrying that too large of chunks are being squished into single tokens and so the model just doesn't have as much time to think per some number of characters." Additionally, each token is seen less often during training, risking under-training of the embedding vectors.

State-of-the-art models typically use vocabulary sizes in the range of ~50K–100K.

```python
# From the GPT model code:
# self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
# self.lm_head = nn.Linear(n_embd, vocab_size)
```

**Answer 20: Extending the Vocabulary of a Pre-Trained Model**

To add new tokens to a pre-trained model:
1. Resize the embedding matrix: add new rows initialized with small random numbers.
2. Resize the LM head: extend the output dimension.
3. Optionally freeze all pre-existing parameters and train only the new embeddings.

Karpathy also mentions an interesting application: "gist tokens" from the paper *Learning to Compress Prompts*. The idea is to introduce a few new tokens and optimize their embeddings so that they replicate the behavior of a very long prompt, providing a form of prompt compression.

---

### Part 10: Tokenization Pitfalls and Practical Advice

**Answer 21: Why LLMs Struggle with Spelling and String Reversal**

Long tokens mean the model sees a multi-character string as one atomic unit. Karpathy demonstrates: "`DefaultCellStyle` turns out to be a single individual token, so that's a lot of characters for a single token." When asked how many L's are in `DefaultCellStyle`, GPT-4 gets it wrong. When asked to reverse it, it produces garbage.

The workaround is to decompose first: "first print out every single character separated by spaces, and then reverse that list." Once the characters are individual tokens, the model can manipulate them.

**Answer 22: Non-English Tokenization Blow-Up**

Karpathy explains: "hello how are you is five tokens and its translation is 15 tokens, so this is a three times blow-up." The tokenizer was trained on predominantly English data, so English text gets longer, more efficient merged tokens. Non-English text is split into many small tokens, consuming more context length. This is "partly the reason that the model works worse on other languages."

**Answer 23: The Trailing-Whitespace Problem**

In GPT-style tokenizers, spaces are typically part of the next word's token (e.g. `" our"` is one token). Adding a trailing space to a prompt creates a bare space token (e.g. token 220 in GPT-2) that the model has almost never seen in isolation. As Karpathy explains: "this space is part of the next token but we're putting it here like this, and the model has seen very very little data of actual space by itself, and we're asking it to complete the sequence. But the problem is that we've sort of begun the first token and now it's been split up and now we're out of distribution."

**Answer 24: The SolidGoldMagikarp Phenomenon**

This is a famous example of a mismatch between the tokenizer's training corpus and the LLM's training corpus. The Reddit user `SolidGoldMagikarp` appeared frequently in the tokenizer training data, earning a dedicated token. But that token never appeared in the LLM's training data, so its embedding row was never updated from random initialization.

Karpathy explains: "in the entire training set for the language model, SolidGoldMagikarp never occurs. That token never appears in the training set for the actual language model. So this token never gets activated — it's initialized at random in the beginning of optimization... it's completely untrained. It's kind of like unallocated memory in a typical binary program written in C."

Invoking such a token feeds random, untrained vectors into the Transformer, producing "undefined behavior" — hallucinations, insults, or safety violations.

**Answer 25: JSON vs YAML Token Efficiency**

Different text formats have different tokenization densities. Karpathy notes: "JSON is actually really dense in tokens and YAML is a lot more efficient. For example, the JSON is 116 and the YAML is 99, so quite a bit of an improvement."

In the "token economy" where you pay per token (both in API cost and in context-length budget), choosing YAML over JSON for structured data can meaningfully reduce costs and allow more content within the context window.

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

json_str = '{"name": "Alice", "age": 30, "city": "Wonderland"}'
yaml_str = 'name: Alice\nage: 30\ncity: Wonderland'

print("JSON tokens:", len(enc.encode(json_str)))
print("YAML tokens:", len(enc.encode(yaml_str)))
```
