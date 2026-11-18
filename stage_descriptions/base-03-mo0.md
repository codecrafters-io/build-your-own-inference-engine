In this stage, you'll implement the `attn` command to mix each token's row with
itself and the rows before it.

### Why attention?

After embedding, each row depends on its token and position. It has no
information from the other tokens yet. In "The capital of France is", the
row for " is" therefore doesn't depend on " France".

Attention combines information from those rows. It assigns weights to positions
and uses them to mix transformed versions of the rows, called values. Causal
attention allows only the current and earlier positions, so a prediction can't
use tokens that come after it.

These attention weights change with the input. The learned weight tensors
loaded from the model file stay fixed.

For example, one of GPT-2's trained attention heads gives the last position
these weights. The values are rounded, so their displayed sum can differ from 1:

| Word | Weight |
| --- | --- |
| "The" | 0.35 |
| "capital" | 0.02 |
| "of" | 0.01 |
| "France" | 0.57 |
| "is" | 0.06 |

More than half of this weighted average comes from the value at "France".
The weights describe how much each value contributes. They don't by themselves
explain what information that value carries.

### Where the weights come from

Attention transforms each input row into three lists of numbers, also called
vectors. A query is compared with keys to decide which positions to use. Each
position's value supplies the information to mix. The model's learned weights
produce these vectors.

A query and key are compared using a dot product: multiply their corresponding
numbers and add the products. A larger result gives that position a higher
score after scaling.
[Softmax](https://nipunbatra.github.io/interactive-articles/softmax-temperature/)
then turns a row's scores into weights that add up to 1. It raises `e`, about
2.718, to each score and divides each result by their total. A score that leads
by 2 gets about 7 times the weight.

One such query/key/value computation is an attention head. GPT-2 runs 12 heads
side by side, and the weights above came from one of them. In this stage,
`attn` runs a single head. Later stages add more heads.

### The `attn` command

`attn` takes three files and a layer number. In this stage's example:

- `config.json` is a JSON object of sizes, using the same names as
  [GPT-2's config](https://huggingface.co/gpt2/blob/main/config.json):

    ```json
    { "n_embd": 1, "n_head": 1, "n_layer": 1, "n_positions": 4, "vocab_size": 2 }
    ```

    `n_embd` is the width of each row, `n_head` the number of attention heads,
    and `n_layer` the number of layers. `n_positions` is the position table's
    size, and `vocab_size` is the number of token IDs in the model.

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
a row of `n` numbers times a weight table of shape `[n, m]` gives `m` numbers,
where number `j` is the dot product of the row with column `j`. A bias is a
learned row added to the result, number by number.

`attn` turns each input row into an output row of the same length. Compute each
position's output row like this:

1. Multiply the position's row by `c_attn.weight` and add `c_attn.bias`. Split
   the result into three equal parts: the query, the key, and the value.
2. Score the position against itself and every earlier position. A score is the
   dot product of this position's query with the other position's key, divided
   by the square root of `n_embd`, the count of numbers in a query.
3. [Softmax](https://nipunbatra.github.io/interactive-articles/softmax-temperature/)
   the scores into weights.
4. Add up the scored positions' values, each multiplied by its weight.
5. Multiply that sum by `c_proj.weight` and add `c_proj.bias`.

### One head, step by step

In this example `n_embd` is 1, so every row, query, key, and value is a single
number, and the input is `[[1], [2]]`. Each step's result is shown under its
number:

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

### Score scaling

<details>
<summary>Why divide the scores by the square root of `n_embd`?</summary>

A dot product sums one product per coordinate. With similarly scaled inputs,
longer queries tend to produce scores with a wider spread. Large score
differences can make softmax concentrate almost all its weight on one position.
Dividing by the square root of the query length keeps typical score magnitudes
more comparable as the width changes.

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

The tester allows a difference of 0.0001 on each number. It will also run two
models whose weights are random numbers between -1 and 1, with rows of four
numbers: one for layer 0, and one that reads layer 1 of a two-layer file. The
numbers change on every run, and the tester's log lists every tensor it wrote.
Every `config.json` in this stage has an `n_head` of 1.

### Notes

- To go deeper on attention, 3Blue1Brown's
  [Attention in transformers](https://www.3blue1brown.com/lessons/attention)
  animates every step, and
  [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  walks through the same vectors in writing.
