In this stage, you'll implement the `attn` command to mix each token's row with
itself and the rows before it.

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

More than half of the new row for "is" comes from "France". GPT-2 often parks
weight it doesn't need on the first word, which is the 0.35 on "The".

### Where the weights come from

Attention turns every row into three lists of numbers. The query is the question
the row asks, the key is what the row says about itself, and the value is what
the row passes on when it matches. The row for "is" might ask "which earlier
word names a place?", and the row for "France" might answer "I name a place."

A score is how well one row's key answers another row's query, and it comes from
the dot product of the two, so the pair above scores highly.
[Softmax](https://nipunbatra.github.io/interactive-articles/softmax-temperature/)
then turns a row's scores into weights that add up to 1, by raising `e` to each
score and dividing by their total. A score that leads by 2 gets about 7 times
the weight.

GPT-2 asks 12 questions at once. It splits each row into 12 slices, called
heads, and each head runs its own weighted average on its slice. The weights for
"is" above came from one head.

### The `attn` command

`attn` takes three files and a layer number. In this stage's example:

- `config.json` is a JSON object of sizes, using the same names as
  [GPT-2's config](https://huggingface.co/gpt2/blob/main/config.json):

    ```json
    { "n_embd": 1, "n_head": 1, "n_layer": 1, "n_positions": 4, "vocab_size": 2 }
    ```

- `model.safetensors` holds four weight tables per layer, named with `h.` and
  the layer number. For layer 0:
    - `h.0.attn.c_attn.weight`, with shape `[n_embd, 3 * n_embd]`
    - `h.0.attn.c_attn.bias`, with shape `[3 * n_embd]`
    - `h.0.attn.c_proj.weight`, with shape `[n_embd, n_embd]`
    - `h.0.attn.c_proj.bias`, with shape `[n_embd]`
- The layer number is the layer to run: `0`. The file holds every layer's
  weights, and `attn` runs one of them.
- `input.json` is a JSON array of the rows to mix, one row per position:
  `[[1], [2]]`.

Multiplying a row by a weight means
[multiplying matrices](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:multiplying-matrices-by-matrices/a/multiplying-matrices):
a row of `n` numbers times a weight of shape `[n, m]` gives `m` numbers, where
number `j` is the dot product of the row with column `j`.

`attn` turns each input row into an output row of the same length. With one
head, compute each position's output row like this:

1. Multiply the position's row by `c_attn.weight` and add `c_attn.bias`. Split
   the result into three equal parts: the query, the key, and the value.
2. Score the position against itself and every earlier position. A score is the
   dot product of this position's query with the other position's key, divided
   by the square root of the head size. The head size is `n_embd / n_head`.
3. [Softmax](https://nipunbatra.github.io/interactive-articles/softmax-temperature/)
   the scores into weights.
4. Add up the scored positions' values, each multiplied by its weight.
5. Multiply that sum by `c_proj.weight` and add `c_proj.bias`.

### One head, step by step

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

### More than one head

With `n_head` heads, split the query, the key, and the value into `n_head` equal
slices. Run steps 2 to 4 on each slice separately, then put the results side by
side, in order, before step 5.

With `n_embd` 2 and 2 heads, each head gets one number. Take the input
`[[1, 0], [0, 1]]`, with weights that copy each row into its query, key, and
value. Here is position 1:

```text
head 0, number 0    scores  0*1 = 0,  0*0 = 0    weights  [0.5, 0.5]
                    value   0.5 * 1 + 0.5 * 0 = 0.5
head 1, number 1    scores  1*0 = 0,  1*1 = 1    weights  [0.269, 0.731]
                    value   0.269 * 0 + 0.731 * 1 = 0.731
side by side        [0.5, 0.731]
```

With one head of size 2 the same rows give `[0.330, 0.670]`, because the softmax
then runs once over the whole row.

### Hints

<details>
<summary>What does the whole algorithm look like as pseudocode?</summary>

```text
// input[i] is the row at position i of input.json, counting from 0

size = n_embd / n_head                // numbers per head

for each position i:
    // 1. query, key, and value, n_embd numbers each
    qkv = input[i] × c_attn.weight + c_attn.bias
    q[i] = qkv[0 : n_embd]
    k[i] = qkv[n_embd : 2 * n_embd]
    v[i] = qkv[2 * n_embd : 3 * n_embd]

for each position i:
    for each head h:
        a, b = h * size, (h + 1) * size   // this head's slice
        for each position j from 0 to i:  // itself and earlier only
            // 2. how well j's key answers i's query
            score[j] = dot(q[i][a : b], k[j][a : b]) / sqrt(size)
        // 3. scores to weights that add up to 1
        weight = softmax(score[0], ..., score[i])
        // 4. the weighted average of the values
        out[i][a : b] = sum over j of weight[j] * v[j][a : b]
    // 5. mix the heads back together
    result[i] = out[i] × c_proj.weight + c_proj.bias
```

</details>

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
- To go deeper on attention, 3Blue1Brown's
  [Attention in transformers](https://www.3blue1brown.com/lessons/attention)
  animates every step, and
  [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  walks through the same vectors in writing.
- To see attention on your own sentences, the
  [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
  runs GPT-2 in the browser and draws each head's weights.
