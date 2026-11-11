In this stage, you'll implement the `embed` command to turn each token id into a row of numbers.

### Embeddings

GPT-2 doesn't work on text directly. It works on tokens, which are words or pieces of words, and each token has a number called its id. GPT-2 knows 50,257 tokens, numbered from 0. For example, the prompt "The capital of France is" is five tokens, with the ids `[464, 3139, 286, 4881, 318]`.

GPT-2 does all its work with arithmetic, so it needs numbers that describe each token. An id can't do that. It's only a label, like a locker number: it tells you which token it is, but nothing about it. So GPT-2 swaps each id for a row of 768 numbers that describe the token, and all of its later steps do their arithmetic on these rows. 768 is just the size GPT-2 uses, and bigger versions of GPT-2 use longer rows.

GPT-2 keeps these rows in a table called `wte.weight`, with one row for each of its 50,257 tokens. Here are the rows for the tokens of "The capital of France is", with the first 4 of their 768 numbers, rounded:

```text
wte.weight     768 columns →
row 0          [ ... ]
...
row 286        [-0.06,  0.02,  0.03,  0.04, ... ]    ← "of"
...
row 318        [-0.01,  0.01,  0.06, -0.11, ... ]    ← "is"
...
row 464        [-0.07, -0.02,  0.06, -0.06, ... ]    ← "The"
...
row 3139       [ 0.06, -0.01,  0.09,  0.06, ... ]    ← "capital"
...
row 4881       [-0.07,  0.05,  0.16, -0.07, ... ]    ← "France"
...
row 50256      [ ... ]
```

The numbers in each row were set during [training](https://developers.google.com/machine-learning/glossary#training), and tokens used in similar ways ended up with similar rows. For example, the rows most similar to the row for Monday are the rows for the other six days of the week.

A token's row says which token it is, but not where it sits in the prompt. Later steps mix the rows together without looking at where they came from, so GPT-2 also adds a row for each token's position. A token's [embedding](https://en.wikipedia.org/wiki/Word_embedding) is the sum of these two rows, added number by number. Both rows come from tables in GPT-2's [weights](https://developers.google.com/machine-learning/glossary#weight):

- `wte.weight` has a row for each token id
- `wpe.weight` has a row for each position a prompt can have (1,024 in GPT-2): row 0 for the first token, row 1 for the second, and so on

For example, "France" is token 4881, and it sits at position 3 in "The capital of France is". Here are the first 4 of its 768 numbers in GPT-2, rounded:

```text
row 4881 of wte.weight      [-0.07,  0.05,  0.16, -0.07, ...]
row 3 of wpe.weight       + [ 0.00, -0.07,  0.11,  0.00, ...]
                            ---------------------------------
embedding of "France"     = [-0.07, -0.02,  0.27, -0.07, ...]
```

### The `embed` Command

`embed` reads two files:

- `model.safetensors`, a weights file like the one `inspect` reads, which holds `wte.weight` and `wpe.weight`
- `ids.json`, a JSON array of token ids, like `[1, 0]`

It prints each token's embedding, in order, as a JSON array of rows.

Each table is a [tensor](https://app.codecrafters.io/concepts/tensors): a flat list of numbers, plus a shape that gives its size. A table's shape is `[rows, columns]`, and its rows are stored one after another, so row `r` starts at index `r * columns`:

```text
values   1, 2, 3, 4, 5, 6        shape [3, 2], so 2 columns
row 0    1, 2                    starts at index 0 * 2 = 0
row 1    3, 4                    starts at index 1 * 2 = 2
row 2    5, 6                    starts at index 2 * 2 = 4
```

{{#lang_is_python}}
The starter code's `load_tensors` returns a dictionary like `{"wte.weight": {"dtype": "F32", "shape": [2, 2], "values": ...}}`. For each tensor, `"values"` holds its numbers and `"dtype"` says what kind they are, where `F32` means 32-bit floats.
{{/lang_is_python}}
{{#lang_is_rust}}
The starter code's `load_tensors` returns a `HashMap<String, Tensor>`, and each `Tensor` holds its shape and its numbers in its `shape` and `values` fields.
{{/lang_is_rust}}
{{#lang_is_go}}
The starter code's `loadTensors` returns a `map[string]tensor`, and each `tensor` holds its shape and its numbers in its `Shape` and `Values` fields.
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
- A slice of `values` is a [`memoryview`](https://docs.python.org/3/library/stdtypes.html#memoryview), which `json.dumps` can't print, so build each embedding as a list. Keep the tensors themselves as the loader returns them, since GPT-2's real weights wouldn't fit in the test server's memory as Python floats.
{{/lang_is_python}}
- To see GPT-2's real embeddings, download its weights as your README's "Running GPT-2" section shows, then run `embed` with `[464, 3139, 286, 4881, 318]` in `ids.json`.
