In this stage, you'll implement the `embed` command to turn each token id into a
row of numbers.

### Why embed?

The token IDs you saw when inspecting the model are labels. An ID's size doesn't
describe the token's meaning. GPT-2 replaces each ID with a learned row of 768
numbers, called a token embedding, that its arithmetic can use. The numbers
were set during [training](https://developers.google.com/machine-learning/glossary#training),
and tokens used in similar ways ended up with similar rows. For example, the six
rows most similar to the row for " Monday" are the other six days of the week.

### Token rows and position rows

The rows are in `wte.weight`, one of GPT-2's learned
[weight tables](https://developers.google.com/machine-learning/glossary#weight).
It has one row per token ID.

A token's row doesn't say where the token sits in the prompt. So a second table,
`wpe.weight`, has one row for each position: row 0 for the first token, row 1
for the second, and so on, up to 1,024 positions.

A position's input embedding is the sum of its token and position rows, added
number by number. For example, " France" is token 4881, including its leading
space. It is at position 3 in "The capital of France is", counting from 0.
Here are the first 4 of its 768 numbers, rounded:

```text
row 4881 of wte.weight      [-0.07,  0.05,  0.16, -0.07, ...]
row 3 of wpe.weight       + [ 0.00, -0.07,  0.11,  0.00, ...]
                            ---------------------------------
embedding of " France"    = [-0.07, -0.02,  0.27, -0.07, ...]
```

If " France" appeared again at position 5, it would use the same row 4881 of
`wte.weight` but row 5 of `wpe.weight`, so its embedding would be different.

### The `embed` command

`embed` reads two files. In this stage's example:

- `ids.json` holds the token ids, `[1, 0]`.
- `model.safetensors` holds the two tables:
    - `wte.weight`, with shape `[2, 2]` and values `1, 2, 3, 4`.
    - `wpe.weight`, with shape `[2, 2]` and values `10, 20, 30, 40`.

For token ID `id` at position `i`, the output row is `wte.weight[id]` plus
`wpe.weight[i]`, added number by number. Both indices count from 0.

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
run two tables of random numbers, with rows of four numbers and ids that repeat
and are not in order. The numbers change on every run, and the tester's log
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
