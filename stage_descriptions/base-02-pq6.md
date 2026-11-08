In this stage, you'll implement the `embed` command to turn each token id into a
row of numbers.

### Why embed?

GPT-2's vocabulary is a fixed list of 50,257 tokens, and a token's id is its
place in that list. For example, "The" is token 464. GPT-2 keeps a table with
one row for each token, and each row is 768 numbers. For "The", it reads row 464
of that table. Later stages work on rows like this one, never on the ids.

### The `embed` command

In `model.safetensors`, that table is the weight table `wte.weight`. A second
weight table, `wpe.weight`, has one row for each position in the prompt. For
each token, this stage adds its row from `wte.weight` to the row for its
position from `wpe.weight`.

For example, take the prompt "The capital of", whose ids are `[464, 3139, 286]`:

| Position | Token | Row from `wte.weight` | Row from `wpe.weight` |
| --- | --- | --- | --- |
| 0 | "The" | row 464 | row 0 |
| 1 | "capital" | row 3139 | row 1 |
| 2 | "of" | row 286 | row 2 |

Each token's embedding is the sum of its two rows. The token row says which word
it is, and the position row says where it sits. If "The" appeared again at
position 5, it would use the same row 464 from `wte.weight` but row 5 from
`wpe.weight`, so its embedding would be different.

`embed` reads two files. In this example:

- `ids.json` holds the token ids, `[1, 0]`.
- `model.safetensors` holds the two weight tables:
    - `wte.weight`, with one row for each token id, `[[1, 2], [3, 4]]`.
    - `wpe.weight`, with one row for each position, `[[10, 20], [30, 40]]`.

A tensor stores its rows one after another. `wte.weight` holds `1, 2, 3, 4` with
shape `[2, 2]`, so row 0 is `[1, 2]` and row 1 is `[3, 4]`.

For each id in `ids.json`, take row `id` of `wte.weight` and row `i` of
`wpe.weight`, where `i` is the id's position, starting at 0. Add the two rows,
number by number:

| Position | Token | Row from `wte.weight` | Row from `wpe.weight` | Embedding |
| --- | --- | --- | --- | --- |
| 0 | 1 | `[3, 4]` | `[10, 20]` | `[13, 24]` |
| 1 | 0 | `[1, 2]` | `[30, 40]` | `[31, 42]` |

Print the embeddings as a JSON array:

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

The tester will also run the same token at two positions, and a sequence whose
ids are not in order.

### Notes

- Print only the JSON to stdout. Send debug output to stderr, the way the
  starter code prints its `Logs from your program will appear here!` line.
{{#lang_is_python}}
- The starter code's `load_tensors` keeps each tensor as a
  [`memoryview`](https://docs.python.org/3/builtins/stdtypes.html#memoryview.cast)
  of the file's bytes. Keep it that way, rather than turning it into a list.
  GPT-2's real weights are 548 MB, and as a list of Python floats they would
  need about 6 GB.
{{/lang_is_python}}
