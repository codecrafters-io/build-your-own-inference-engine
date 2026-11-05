In this stage, you'll reuse past keys and values in `generate` instead of
recomputing the whole sequence.

### Why keep keys and values?

Each new token only needs its own query. The keys and values of the tokens
already written do not change. GPT-2 has 12 layers, and scoring a new token by
running every earlier token through all 12 layers repeats that work. Keeping the
keys and values makes the new ids come out the same, with less arithmetic.

That saving matters when you run GPT-2's real weights in a later stage. Each
token there passes through 124 million numbers, and an engine that recomputes
the whole sequence at every step can run out of time.

### The `generate` command

The command is the same one as greedy decoding. `tokens.json` again holds a
count, and there is no temperature. The ids must match a run that recomputes the
whole sequence at every step.

For each layer, and for each head, keep the keys and the values you have already
computed. A key is the key block of that head after the `c_attn` projection, and
a value is the value block. They are the vectors from step 1 of attention,
before the scores.

To add a token, run the block on that token's row only. Append its key and its
value to the lists for each head. Its scores use the new query against every
stored key, including its own. The softmax and the output projection are
unchanged. The layer norms and the MLP work on one row at a time, so they only
need the new row.

In this example the prompt is `[0]` and `tokens.json` is `{ "count": 3 }`.
`ln_f.weight` is `[0, 0]` and `ln_f.bias` is `[0, 1]`, so token 1 has the
highest score at every step. The ids are:

```json
[1, 1, 1]
```

### Tests

The tester will write the four files and execute your program like this:

```bash
$ ./your_program.sh generate model.safetensors config.json ids.json tokens.json
```

Your program must print:

```json
[1, 1, 1]
```

The ids have to match exactly.

### Notes

- Store keys and values after `c_attn`, split into heads, and before the
  softmax.
- The new token's position is still its index in the whole sequence. The cache
  does not change the position row you add.
- A prompt is cached in order, left to right, before the first new token.
