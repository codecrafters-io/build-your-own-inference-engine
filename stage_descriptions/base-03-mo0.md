In this stage, you'll implement the `attn` command to mix each token's row with
itself and the rows before it.

### Why attention?

After embedding, the row for a token is only that token: in "The capital of
France is", nothing in the row for "is" says which capital comes next. Attention
replaces each row with a weighted average of itself and the rows before it. GPT-2
writes left to right, so later rows don't exist yet. In one of its later layers,
GPT-2 gives the row for "is" these weights:

| Word | Weight |
| --- | --- |
| "The" | 0.347 |
| "capital" | 0.017 |
| "of" | 0.007 |
| "France" | 0.571 |
| "is" | 0.058 |

More than half of the new row for "is" comes from "France".

### Where the weights come from

Attention turns every row into three lists of numbers: the query is the question
the row asks, the key is what it says about itself, and the value is what it
passes on when it matches. The row for "is" might ask "which earlier word names
a place?", and "France" answers "I name a place."

A score is how well one row's key answers another row's query: the dot product
of the two, divided by the square root of `n_embd`. Longer queries give larger
dot products, and large scores make softmax put almost all the weight on one
position, so dividing by the square root of the length keeps scores in the same
range.

[Softmax](https://nipunbatra.github.io/interactive-articles/softmax-temperature/)
then turns a row's scores into weights that add up to 1, by raising `e` to each
score and dividing by their total. A score that leads by 2 gets about 7 times
the weight.

A query, a key, a value, and their weights make up one head. GPT-2 runs 12 side
by side, and the weights above came from one of them. `attn` runs a single head.
Later stages add the rest.

### The `attn` command

`attn` takes three files and a layer number:

- `config.json` holds the sizes, named as in
  [GPT-2's config](https://huggingface.co/gpt2/blob/main/config.json):

    ```json
    { "n_embd": 1, "n_head": 1, "n_layer": 1, "n_positions": 4, "vocab_size": 2 }
    ```

- `model.safetensors` holds four weight tables per layer in this stage's files,
  named with `h.` and the layer number. For layer 0:
    - `h.0.attn.c_attn.weight`, with shape `[n_embd, 3 * n_embd]`
    - `h.0.attn.c_attn.bias`, with shape `[3 * n_embd]`
    - `h.0.attn.c_proj.weight`, with shape `[n_embd, n_embd]`
    - `h.0.attn.c_proj.bias`, with shape `[n_embd]`
- The layer number is the layer to run: `0`. The file holds every layer, and
  `attn` runs one.
- `input.json` is one row per position: `[[1], [2]]`.

Multiplying a row by a weight is
[matrix multiplication](https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:multiplying-matrices-by-matrices/a/multiplying-matrices):
a row of `n` numbers times a weight of shape `[n, m]` gives `m` numbers, number
`j` being the dot product of the row with column `j`.

Compute each position's output row like this:

1. Multiply the row by `c_attn.weight`, add `c_attn.bias`, and split the result
   into three equal parts: the query, the key, and the value.
2. Score the position against itself and every earlier position, as above, with
   this position's query and the other position's key.
3. [Softmax](https://nipunbatra.github.io/interactive-articles/softmax-temperature/)
   the scores into weights.
4. Add up the scored positions' values, each multiplied by its weight.
5. Multiply that sum by `c_proj.weight` and add `c_proj.bias`.

### One head, step by step

Here `n_embd` is 1, so every row, query, key, and value is a single number:

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

### Tests

The tester writes the three files from the example, then runs:

```bash
$ ./your_program.sh attn model.safetensors config.json 0 input.json
```

Your program must print:

```json
[[1.0], [1.8807970779778822]]
```

The tester allows a difference of 0.0001 on each number. It will also run two
random-weight models with rows of four numbers, the second reading layer 1 of a
two-layer file. Every `config.json` here has an `n_head` of 1.
