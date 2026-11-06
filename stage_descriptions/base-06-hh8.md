In this stage, you'll implement the `generate` command to repeatedly take the
highest-scoring token.

### Why greedy?

Logits score every possible next token, but writing text means choosing one.
Greedy decoding is the simplest choice. It takes the highest score, appends that
token, and scores the longer sequence again. It uses no randomness, so the same
prompt always continues the same way, and you can check every id it writes.

Here is GPT-2 continuing "The capital of France is", one step at a time:

| Step | Picked | Runner-up | Lead |
| --- | --- | --- | --- |
| 1 | " the" | " now" | 0.57 |
| 2 | " capital" | " city" | 1.62 |
| 3 | " of" | "," | 3.87 |
| 4 | " the" | " France" | 0.33 |
| 5 | " French" | " Republic" | 0.96 |
| 6 | " Republic" | " Empire" | 1.32 |

At step 4, a lead of 0.33 is all that separates " the capital of the French
Republic" from " the capital of France".

### The `generate` command

`config.json` and `ids.json` are the same format as for `logits`.
`tokens.json` is a JSON object. In this stage it holds the count:

```json
{ "count": 2 }
```

Generate the new ids like this:

```text
repeat count times:
    scores = the logits for ids, as logits computes them
    best = the id with the highest score, the lower id on a tie
    append best to ids, and to the new ids
print the new ids as a JSON array
```

In this example `ln_f.bias` is `[0, 1]` and `ln_f.weight` is `[0, 0]`, so the
normalized last row is always `[0, 1]`. Against the token rows `[1, 0]` and
`[0, 1]`, token 0's score is 0 and token 1's score is 1. Both new tokens are
token 1:

```json
[1, 1]
```

### Hints

<details>
<summary>How do I keep the lower id on a tie?</summary>

Go through the ids from 0 upward, and replace the best id only when a score is
strictly greater than the best so far. An equal score then keeps the earlier,
lower id.

</details>

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

- The new token's position is its index in the whole sequence. A prompt of
  length 3 puts the first new token at position 3.
