In this stage, you'll extend the `attn` command to split each row into slices
that each get their own weights.

### Why more than one head?

Your `attn` gives each row one set of weights. For example, take the sentence
"The capital of France is". More than half of the weight for "is" went to
"France", which leaves little for anything else the row might need.

GPT-2 looks in 12 ways at once. It splits each query, key, and value into 12
slices, called heads, and each head works out its own weights from its own
slice. Here are two of GPT-2's heads for "is", in the same layer:

| Word | One head | Another head |
| --- | --- | --- |
| "The" | 0.35 | 0.63 |
| "capital" | 0.02 | 0.24 |
| "of" | 0.01 | 0.01 |
| "France" | 0.57 | 0.05 |
| "is" | 0.06 | 0.07 |

The first head is the one you saw before. The second puts about a quarter of its
weight on "capital" and almost none on "France". Each head takes its own
weighted average, and the results go side by side to make the new row.

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
softmax then runs once over both numbers.

### Hints

<details>
<summary>What does the whole algorithm look like as pseudocode?</summary>

```text
size = n_embd / n_head                    // numbers per head

// q[i], k[i], and v[i] are position i's query, key, and value, as before

for each position i:
    for each head h:
        a, b = h * size, (h + 1) * size   // this head's slice
        for each position j from 0 to i:
            score[j] = dot(q[i][a : b], k[j][a : b]) / sqrt(size)
        weight = softmax(score[0], ..., score[i])
        out[i][a : b] = sum over j of weight[j] * v[j][a : b]
    result[i] = out[i] × c_proj.weight + c_proj.bias
```

</details>

<details>
<summary>Why does the division change to the head size?</summary>

The division keeps scores in the same range whatever the length of the query. A
head's score now adds up only the products in its slice, so the length that
matters is the slice's. GPT-2's heads have 64 numbers, so it divides by 8.

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
