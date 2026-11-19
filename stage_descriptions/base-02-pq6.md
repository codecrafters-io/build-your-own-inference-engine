In this stage, you'll implement the `embed` command to turn each token id into a
row of numbers. It's the first step of generation, since the layers do their
arithmetic on rows, never on ids.

### Why embed?

GPT-2 doesn't read text. It reads tokens, which are words or pieces of words,
and each token has a number called its id. GPT-2 knows 50,257 tokens. For
example, the prompt "The capital of France is" is five tokens, with the ids
`[464, 3139, 286, 4881, 318]`.

An id is only a label, like a locker number. It says which token this is, and
nothing about it. GPT-2 does all its work with arithmetic, so it swaps each id
for a row of 768 numbers that describe the token. Those numbers were set during
[training](https://developers.google.com/machine-learning/glossary#training),
and tokens used in similar ways ended up with similar rows. For example, the six
rows most similar to the row for " Monday" are the other six days of the week.

### Token rows and position rows

The rows are in a table called `wte.weight`, with one row for each token id. It
is one of GPT-2's
[weights](https://developers.google.com/machine-learning/glossary#weight), the
numbers it learned in training.

A token's row doesn't say where the token sits in the prompt. So a second table,
`wpe.weight`, has one row for each position: row 0 for the first token, row 1
for the second, and so on, up to 1,024 positions.

A token's embedding is the sum of its two rows, added number by number. For
example, "France" is token 4881, at position 3 in "The capital of France is".
Here are the first 4 of its 768 numbers, rounded:

```text
row 4881 of wte.weight      [-0.07,  0.05,  0.16, -0.07, ...]
row 3 of wpe.weight       + [ 0.00, -0.07,  0.11,  0.00, ...]
                            ---------------------------------
embedding of "France"     = [-0.07, -0.02,  0.27, -0.07, ...]
```

If "France" appeared again at position 5, it would use the same row 4881 of
`wte.weight` but row 5 of `wpe.weight`, so its embedding would be different.

### The `embed` command

`embed` reads two files. In this stage's example:

- `ids.json` holds the token ids, `[1, 0]`.
- `model.safetensors` holds the two tables:
    - `wte.weight`, with shape `[2, 2]` and values `1, 2, 3, 4`.
    - `wpe.weight`, with shape `[2, 2]` and values `10, 20, 30, 40`.

For the id at position `i` in `ids.json`, counting from 0:

1. Take row `id` of `wte.weight`.
2. Take row `i` of `wpe.weight`.
3. Add the two rows, number by number.

### Finding a row

The loader gives you each table as a
[tensor](https://app.codecrafters.io/concepts/tensors): one flat list of its
numbers, and a shape that says how to cut the list into rows. With a shape of
`[rows, columns]`, the rows sit one after another, so row `r` starts at index
`r * columns`:

```text
wte.weight, shape [2, 2]

index      0    1    2    3
values     1    2    3    4
           ------   ------
           row 0    row 1
```

{{#lang_is_python}}
The starter code's `load_tensors` returns a dictionary like
`{"wte.weight": {"dtype": "F32", "shape": [2, 2], "values": ...}}`. For each
tensor, `"values"` holds its flat list of numbers.
{{/lang_is_python}}
{{#lang_is_rust}}
The starter code's `load_tensors` returns a `HashMap<String, Tensor>`. Each
`Tensor` holds its shape in `shape` and its flat list of numbers in `values`.
{{/lang_is_rust}}
{{#lang_is_go}}
The starter code's `loadTensors` returns a `map[string]tensor`. Each `tensor`
holds its shape in `Shape` and its flat list of numbers in `Values`.
{{/lang_is_go}}

So in this example, the rows of `wte.weight` are `[1, 2]` and `[3, 4]`, and the
rows of `wpe.weight` are `[10, 20]` and `[30, 40]`:

| Position | Id | Row of `wte.weight` | Row of `wpe.weight` | Embedding |
| --- | --- | --- | --- | --- |
| 0 | 1 | `[3, 4]` | `[10, 20]` | `[13, 24]` |
| 1 | 0 | `[1, 2]` | `[30, 40]` | `[31, 42]` |

Print the embeddings in order, as a JSON array of rows:

```json
[
  [13, 24],
  [31, 42]
]
```

### Tests

The tester will write `model.safetensors` and `ids.json`, then execute your
program like this:

```bash
$ ./your_program.sh embed model.safetensors ids.json
```

Your program must print:

```json
[
  [13.0, 24.0],
  [31.0, 42.0]
]
```

The tester reads your output as JSON, so `13` and `13.0` both pass. It will also
run two tables of random numbers, with rows of four numbers and ids that are
not sorted and may repeat. The numbers change on every run, and the tester's log
lists every tensor it wrote.

### Notes

- Print only the JSON to stdout. Send debug output to stderr, the way the
  starter code prints its `Logs from your program will appear here!` line.
{{#lang_is_python}}
- A slice of `values` is a
  [`memoryview`](https://docs.python.org/3/builtins/stdtypes.html#memoryview),
  which `json.dumps` can't print, so build each embedding as a list. Keep the
  tensors themselves as the loader returns them. GPT-2's real weights wouldn't
  fit in the test server's memory as Python floats.
{{/lang_is_python}}
