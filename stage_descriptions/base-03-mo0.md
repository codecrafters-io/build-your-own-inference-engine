In this stage, you'll implement the `attn` command to mix each token's row with
the rows of the tokens before it.

### Why attention?

After embedding, the row for a token is only that token. For example,
take the sentence "The capital of France is". The row for "is" does not yet
depend on "France". Attention is the step that mixes earlier rows into the
current one. "is" looks at "The", "capital", "of", "France", and "is", and its
output row depends on all five. It does not look at a word that comes after it,
because that word has not been written yet.

GPT-2 runs this mix 12 times on each row, once per head. Each head uses 64 of
the 768 numbers, and divides its scores by 8, the square root of 64. `n_head` is
12 in [GPT-2's config](https://huggingface.co/gpt2/blob/main/config.json).

### The `attn` command

The tester writes four files. `config.json` uses the same names as GPT-2:

```json
{ "n_embd": 1, "n_head": 1, "n_layer": 1, "n_positions": 4, "vocab_size": 2 }
```

`layer.json` is the layer number, `0` in this example. `input.json` is a JSON
array of rows, one row per position. This example's input is `[[1], [2]]`.

The layer's attention weights are `h.<layer>.attn.c_attn.weight`,
`h.<layer>.attn.c_attn.bias`, `h.<layer>.attn.c_proj.weight`, and
`h.<layer>.attn.c_proj.bias`.

1. Multiply each row by `c_attn.weight` and add `c_attn.bias`. The result is
   three blocks side by side: query, key, and value. Each block has `n_embd`
   numbers.
2. Split each block into `n_head` heads. The head size is `n_embd / n_head`.
3. A score is the dot product of a query and a key, divided by the square root
   of the head size. This file's head size is 1, so you divide by 1.
4. A position may only use itself and earlier positions. Drop the later scores
   before the softmax.
5. Softmax the scores that remain. Softmax turns scores into weights: raise `e`
   to each score, then divide each result by the sum of all of them. The
   weighted sum of the values is that head's output.
6. Join the heads, multiply by `c_proj.weight`, and add `c_proj.bias`.

In this file the weights copy the input into the query, the key, and the value,
and the projection puts that result back unchanged. Position 0's only score is
`1 * 1`. The softmax of one score is 1, so the output is `[1]`. Position 1's
scores are 2 and 4. Their softmax weights are about 0.119 and 0.881, and
`0.119 * 1 + 0.881 * 2` is about 1.881.

Print one JSON array of rows:

```json
[[1.0], [1.8807970779778822]]
```

### Tests

The tester will write the four files and execute your program like this:

```bash
$ ./your_program.sh attn model.safetensors config.json layer.json input.json
```

Your program must print:

```json
[[1.0], [1.8807970779778822]]
```

The tester allows a difference of 0.0001 on each number. It will also run a head
of size 2, where forgetting the division changes the second row, a model with
two heads, and layer 1.

### Notes

- Read `n_embd` and `n_head` from `config.json`. The head size is
  `n_embd / n_head`.
- Softmax runs over the positions a token is allowed to see, and the weights for
  one query add up to 1.
