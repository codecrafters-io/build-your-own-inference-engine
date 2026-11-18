In this stage, you'll extend `attn` to use multiple heads, each with its own
attention weights over the input positions.

### Why more than one head?

Your `attn` gives each position one set of weights over the positions it can
use. Multiple heads allow different sets of weights at once. For "The capital
of France is", one head can emphasize "France" while another emphasizes
different parts of the same input.

GPT-2 looks in 12 ways at once. It splits each query, key, and value into 12
slices, called heads, and each head works out its own weights from its own
slice. Here are two of GPT-2's heads for "is", in the same layer, with weights
rounded to two decimal places:

| Word | One head | Another head |
| --- | --- | --- |
| "The" | 0.35 | 0.63 |
| "capital" | 0.02 | 0.24 |
| "of" | 0.01 | 0.01 |
| "France" | 0.57 | 0.05 |
| "is" | 0.06 | 0.07 |

The first head is the one you saw when learning single-head attention. The
second puts about a quarter of its weight on "capital". Each head mixes its
own value slices, and the results go side by side before the output projection.

### Splitting into heads

`config.json` says how many heads there are, in `n_head`. The head size is
`n_embd / n_head`, the count of numbers in each slice. GPT-2 has 768 numbers in
a row and 12 heads, so its head size is 64.

The query, key, and value are worked out as before, and so is the multiply by
`c_proj.weight` at the end. The steps between them change:

1. Split the query, the key, and the value into `n_head` equal slices, in order.
   Head 0 gets the first `n_embd / n_head` numbers of each, head 1 the next, and
   so on.
2. For each head, score the position against itself and every earlier position,
   using only that head's slices. Divide each score by the square root of the
   head size, where you divided by the square root of `n_embd` before.
3. For each head, softmax its scores into weights, and add up its slices of the
   values, each multiplied by its weight.
4. Put the heads' results side by side, in head order. That gives `n_embd`
   numbers again, ready for `c_proj.weight`.

### Two heads, step by step

In this example `n_embd` is 2 and `n_head` is 2, so each head gets one number.
The weights copy each row into its query, key, and value, and the input is
`[[1, 0], [0, 1]]`. Here is position 1, whose query is `[0, 1]`:

```text
// head 0 uses number 0 of each query, key, and value
scores        0*1 = 0,  0*0 = 0            // / sqrt(1)
weights       [0.5, 0.5]
average       0.5 * 1 + 0.5 * 0 = 0.5

// head 1 uses number 1
scores        1*0 = 0,  1*1 = 1
weights       [0.269, 0.731]
average       0.269 * 0 + 0.731 * 1 = 0.731

// side by side, then × c_proj.weight, which leaves them unchanged
output row    [0.5, 0.731]
```

Position 0 has only itself to look at, so each head gives it a weight of 1 and
its row stays `[1, 0]`. Print one JSON array of rows:

```json
[[1.0, 0.0], [0.5, 0.7310585786300049]]
```

With one head, the same files give `[0.330, 0.670]` for position 1, because the
query/key dot products use both coordinates and the divisor becomes `sqrt(2)`.
Softmax still operates on scores for positions. With two heads, each head has
its own scores and softmax.

### Scaling within a head

<details>
<summary>Why does the division change to the head size?</summary>

Scaling accounts for how many products contribute to a dot product. A head's
score now adds up only the products in its slice, so the relevant length is
the head size. GPT-2's heads have 64 numbers, so each score is divided by 8.

</details>

### Tests

The tester will write the three files from the example, then execute your
program like this:

```bash
$ ./your_program.sh attn model.safetensors config.json 0 input.json
```

Your program must print:

```json
[[1.0, 0.0], [0.5, 0.7310585786300049]]
```

The tester allows a difference of 0.0001 on each number. It will also run two
models whose weights are random numbers between -1 and 1, with two heads and
rows of four numbers: one for layer 0, and one that reads layer 1 of a two-layer
file. The numbers change on every run, and the tester's log lists every tensor
it wrote.

### Notes

- With an `n_head` of 1 the only slice is the whole row, so the files your
  `attn` already handles give the same output.
- To see heads on your own sentences, the
  [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
  runs GPT-2 in the browser and draws each head's weights.
