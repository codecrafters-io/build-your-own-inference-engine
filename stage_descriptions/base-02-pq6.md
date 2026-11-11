In this stage, you'll implement the `embed` command to turn each token id into a row of numbers.

### Embeddings

GPT-2 doesn't read text. It reads tokens, which are words or pieces of words, and each token has a number called its id. For example, the prompt "The capital of France is" is five tokens, with the ids `[464, 3139, 286, 4881, 318]`.

An id is only a label. Token 4881 isn't bigger than token 318 in any useful way, so GPT-2 starts by swapping each id for a row of 768 numbers, and everything it does after that works on these rows. GPT-2 learned the rows during training, and tokens used in similar ways ended up with similar rows. For example, in GPT-2's table, the rows most similar to the row for Monday, as in "See you on Monday", are the rows for the other six days of the week.

A token's row says which token it is, but not where it sits. Later steps mix the rows together, so GPT-2 also adds a row for each token's position. A token's embedding is the sum of these two rows, added number by number:

- Its row from `wte.weight`, which has a row for each token id
- The row for its position from `wpe.weight`, which has a row for each position a prompt can have (1,024 in GPT-2): row 0 for the first token, row 1 for the second, and so on

### The `embed` Command

`embed` reads two files: `model.safetensors`, which holds `wte.weight` and `wpe.weight`, and `ids.json`, a JSON array of token ids like `[1, 0]`. It prints each token's embedding, in order, as a JSON array of rows.

Each table is a [tensor](https://app.codecrafters.io/concepts/tensors): a flat list of numbers, plus a shape that says how to split it into rows. In a tensor with `c` numbers per row, row `r` is the `c` numbers starting at index `r * c`:

```text
values   1, 2, 3, 4, 5, 6        shape [3, 2], so c is 2
row 0    1, 2                    starts at index 0 * 2 = 0
row 1    3, 4                    starts at index 1 * 2 = 2
row 2    5, 6                    starts at index 2 * 2 = 4
```

{{#lang_is_python}}
The starter code's `load_tensors` returns a dictionary from each tensor's name to its `"shape"` and its numbers, `"values"`.
{{/lang_is_python}}
{{#lang_is_rust}}
The starter code's `load_tensors` returns a `HashMap` from each tensor's name to a `Tensor`, whose `shape` and `values` fields hold its shape and its numbers.
{{/lang_is_rust}}
{{#lang_is_go}}
The starter code's `loadTensors` returns a map from each tensor's name to a `tensor`, whose `Shape` and `Values` fields hold its shape and its numbers.
{{/lang_is_go}}

For example, take two tiny tables with 2 numbers per row:

- `wte.weight`, with shape `[2, 2]` and values `1, 2, 3, 4`
- `wpe.weight`, with shape `[2, 2]` and values `10, 20, 30, 40`

With the ids `[1, 0]`, the embeddings are:

| Position | Id | `wte.weight` row | `wpe.weight` row | Embedding |
| --- | --- | --- | --- | --- |
| 0 | 1 | `[3, 4]` | `[10, 20]` | `[13, 24]` |
| 1 | 0 | `[1, 2]` | `[30, 40]` | `[31, 42]` |

### Tests

The tester will write the example's `model.safetensors` and `ids.json`, then execute your program like this:

```bash
$ ./your_program.sh embed model.safetensors ids.json
```

Your program must print:

```json
[[13.0, 24.0], [31.0, 42.0]]
```

The tester reads your output as JSON numbers, so `13` and `13.0` both pass. It will also run a prompt that repeats an id at two positions, and a prompt with three tokens.

### Notes

{{#lang_is_python}}
- A slice of `values` is a `memoryview`, which `json.dumps` can't print, so build each embedding as a list. Keep the tensors themselves as the loader returns them, since GPT-2's real weights wouldn't fit in memory as Python floats.
{{/lang_is_python}}
- To see GPT-2's real embeddings, download its weights as your README's "Running GPT-2" section shows, then run `embed` with `[464, 3139, 286, 4881, 318]` in `ids.json`.
- 3Blue1Brown's [Transformers, the tech behind LLMs](https://www.3blue1brown.com/lessons/gpt) shows what embeddings capture, with pictures.
