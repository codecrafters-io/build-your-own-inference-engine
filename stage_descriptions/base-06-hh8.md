In this stage, you'll implement the `generate` command to repeatedly take the
highest-scoring token.

### Why greedy?

Logits score every possible next token, but writing text means choosing one.
Greedy decoding is the simplest choice. It takes the highest score, appends that
token, and scores the longer sequence again. It uses no randomness, so the same
prompt always continues the same way, and you can check every id it writes.

### The `generate` command

`config.json` and `ids.json` are the same format as for `logits`.
`tokens.json` is a JSON object. In this stage it holds the count:

```json
{ "count": 2 }
```

Compute the logits for the ids. Append the id with the highest score. When two
scores are equal, keep the lower id. Repeat until you have appended `count`
tokens. Print only the new ids, as a JSON array.

In this example `ln_f.bias` is `[0, 1]` and `ln_f.weight` is `[0, 0]`, so the
normalized last row is always `[0, 1]`. Against the token rows `[1, 0]` and
`[0, 1]`, token 0's score is 0 and token 1's score is 1. Both new tokens are
token 1:

```json
[1, 1]
```

### Tests

The tester will write the four files and execute your program like this:

```bash
$ ./your_program.sh generate model.safetensors config.json ids.json tokens.json
```

Your program must print:

```json
[1, 1]
```

The ids have to match exactly. The tester will also run a file where every token
has the same score, so the lower id, 0, is chosen twice.

### Notes

- A tie keeps the lower token id. Compare scores with a strict greater-than.
- The new token's position is its index in the whole sequence. A prompt of
  length 3 puts the first new token at position 3.
- Print the new ids only. The prompt stays in the input file.
