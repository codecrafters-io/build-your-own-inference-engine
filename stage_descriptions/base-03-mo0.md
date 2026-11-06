In this stage, you'll implement the `attn` command to mix each token's row with
the rows of the tokens before it.

### Why attention?

After embedding, the row for a token is only that token. For example, take the
sentence "The capital of France is". The row for "is" does not yet depend on
"France", so nothing in it says which capital comes next.

Attention fixes this by replacing each row with a weighted average of itself and
the rows before it. Later words don't exist yet when GPT-2 writes a word. For
example, in one of its later steps, GPT-2 gives the row for "is" these weights:

| Word | Weight |
| --- | --- |
| "The" | 0.35 |
| "capital" | 0.02 |
| "of" | 0.01 |
| "France" | 0.57 |
| "is" | 0.06 |

More than half of the new row for "is" comes from "France". Most of the rest
goes to "The", because GPT-2 often puts weight it doesn't need on the first
word.

### Where the weights come from

Each row asks a question of the rows before it, and each of those rows answers.
Attention turns every row into three lists of numbers:

- The query is the question the row asks. The row for "is" might ask "which
  earlier word names a place?"
- The key is what the row says about itself. The row for "France" might say
  "I name a place."
- The value is what the row passes on when it matches. The row for "France"
  might pass on "a country in Europe."

A score is how well one row's key answers another row's query. It's the dot
product of the two, so a bigger score means a better match. The key for
"France" answers the query for "is" well, so that pair gets a high score.

Softmax turns a row's scores into weights that add up to 1. It raises `e`, about
2.718, to each score, which makes every number positive and lets a bigger score
count for much more. Then it divides each result by their total:

```text
scores     2       4
e^score    7.39    54.60      total 61.99
weights    0.119   0.881      7.39 / 61.99 and 54.60 / 61.99
```

A score that leads by 2 gets about 7 times the weight.

GPT-2 asks 12 questions at once. It splits each row into 12 slices, called
heads, and each head runs its own weighted average on its slice. The weights for
"is" above came from one head. Another head asks "which word came just before
me?", and puts almost all of each word's weight on that word:

```text
"capital"  →  "The"
"of"       →  "capital"
"France"   →  "of"
"is"       →  "France"
```

"The" has no word before it, so it uses itself.

### Multiplying by a weight

The steps below use two operations on lists of numbers. The dot product of two
lists of the same length multiplies them pair by pair and adds up the results.
Multiplying a row by a matrix is a loop of dot products. If the row has `n`
numbers and the matrix has shape `[n, m]`, the result has `m` numbers, and
number `j` is the dot product of the row with column `j` of the matrix:

```text
row     [1, 2]
matrix  [[1, 0, 2],
         [3, 1, 0]]
result  [1*1 + 2*3, 1*0 + 2*1, 1*2 + 2*0] = [7, 2, 2]
```

Whenever this stage or a later one says to multiply by a weight, it means this.

### The `attn` command

`attn` takes three files and a layer number. In this stage's example:

- `model.safetensors` is a safetensors file, like the one `inspect` reads. It
  holds four weight tables for each layer. Their names start with `h.` and the
  layer number, so for layer 0 they are:
    - `h.0.attn.c_attn.weight`, with shape `[n_embd, 3 * n_embd]`
    - `h.0.attn.c_attn.bias`, with shape `[3 * n_embd]`
    - `h.0.attn.c_proj.weight`, with shape `[n_embd, n_embd]`
    - `h.0.attn.c_proj.bias`, with shape `[n_embd]`
- `config.json` is a JSON object of sizes, using the same names as
  [GPT-2's config](https://huggingface.co/gpt2/blob/main/config.json):

    ```json
    { "n_embd": 1, "n_head": 1, "n_layer": 1, "n_positions": 4, "vocab_size": 2 }
    ```

- The layer number is the layer to run: `0`. `model.safetensors` holds the
  weights for every layer, and `attn` runs only one of them.
- `input.json` is a JSON array of the rows to mix, one row per position:
  `[[1], [2]]`.

`attn` turns each input row into an output row of the same length, mixed from
itself and the rows before it as described above. Print the output rows as a
JSON array. Compute them like this:

```text
size = n_embd / n_head                    // the head size

for each position i:
    // 1. query, key, and value: n_embd numbers in, 3 * n_embd out
    q[i], k[i], v[i] = thirds of (row[i] × c_attn.weight + c_attn.bias)

// head h uses numbers h * size through (h + 1) * size - 1 of q, k, and v
for each head h:
    for each position i:
        for each position j from 0 to i:  // itself and earlier positions only
            // 2. how well j's key answers i's query
            score[j] = dot(q[i] in head h, k[j] in head h) / sqrt(size)
        // 3. scores to weights that add up to 1
        weight = softmax(score[0], ..., score[i])
        // 4. the weighted average of the values
        out[i] in head h = sum over j of weight[j] * (v[j] in head h)

for each position i:
    // 5. mix the heads back together
    result[i] = out[i] × c_proj.weight + c_proj.bias
```

### Example

This example has one head of size 1, and the input is `[[1], [2]]`. Each step's
result is shown under its number:

```text
// 1. row × c_attn.weight [[1, 1, 1]] + c_attn.bias [0, 0, 0]
position 0    query 1   key 1   value 1
position 1    query 2   key 2   value 2

// 2. query · key / sqrt(1), skipping later positions
position 0    1*1 = 1
position 1    2*1 = 2,  2*2 = 4

// 3. softmax
position 0    [1]                  // a single score always gets weight 1
position 1    [0.119, 0.881]

// 4. weighted average of the values
position 0    1 * 1 = 1
position 1    0.119 * 1 + 0.881 * 2 = 1.881

// 5. × c_proj.weight [[1]] + c_proj.bias [0] leaves both unchanged
output        [[1], [1.881]]
```

Print one JSON array of rows:

```json
[[1.0], [1.8807970779778822]]
```

### Hints

<details>
<summary>Why divide the scores by the square root of the head size?</summary>

A dot product adds one product per number, so wider heads give larger scores.
Large scores make the softmax put almost all the weight on one position.
Dividing by the square root of the head size keeps the scores in the same range
whatever the width. GPT-2's heads have 64 numbers, so it divides by 8.

</details>

### Tests

The tester will write the three files from the example, then execute your
program like this:

```bash
$ ./your_program.sh attn model.safetensors config.json 0 input.json
```

Your program must print:

```json
[[1.0], [1.8807970779778822]]
```

The tester allows a difference of 0.0001 on each number. It will also run a head
of size 2, where forgetting the division changes the second row, a model with
two heads, and layer 1.

### Notes

- With GPT-2's real weights, scores can be large enough that raising `e` to them
  overflows. Subtract the largest score from every score first. The weights come
  out the same, because softmax only depends on the differences between scores.
- To see attention on your own sentences, the
  [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
  runs GPT-2 in the browser and draws each head's weights.
