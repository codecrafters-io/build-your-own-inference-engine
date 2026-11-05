In this stage, you'll implement the `embed` command to add token embeddings and
position embeddings.

### Why embed?

GPT-2 doesn't read words. It reads token ids, integers that each stand for a
word or a piece of one. For example, the word "The" is token 464. Later stages
multiply numbers, so each id first has to become a row of numbers. The row you
look up for a token is called its embedding.

GPT-2 stores two tables for this. `wte.weight` is the token embedding table,
with one row per token id. `wpe.weight` is the position embedding table, with
one row per position in the prompt. For each token, this stage adds its row from
`wte.weight` to the row for its position from `wpe.weight`.

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

### The `embed` command

`ids.json` is a JSON array of token ids, like `[1, 0]`. For each id at position
`i`, starting at 0, add row `id` of `wte.weight` to row `i` of `wpe.weight`.
Both tables are `F32`.

In this example, `wte.weight` is `[[1, 2], [3, 4]]` and `wpe.weight` is
`[[10, 20], [30, 40]]`. The ids are `[1, 0]`.

Position 0 is token 1, so `[3, 4] + [10, 20]` is `[13, 24]`. Position 1 is token
0, so `[1, 2] + [30, 40]` is `[31, 42]`.

Print a JSON array with one row per token:

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

The tester allows a difference of 0.00001 on each number. It will also run the
same token at two positions, and a sequence whose ids are not in order.

### Notes

- Position 0 is the first id. There is no separate list of positions.
- The two rows you add have the same length. Print one JSON array to stdout, and
  send debug output to stderr.
{{#lang_is_python}}
- We recommend keeping each tensor as a `memoryview` of the file's bytes, cast
  to floats, rather than a list. A later stage loads GPT-2's real weights, 548
  MB of them, and a list of Python floats would need about 6 GB:

    ```python
    values = memoryview(data)[start:end].cast("f")
    ```
{{/lang_is_python}}
