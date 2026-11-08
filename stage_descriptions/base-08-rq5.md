In this stage, you'll reuse past keys and values in `generate` instead of
recomputing the whole sequence.

### Why keep keys and values?

A new token needs its own query, key, and value. The keys and values of the
tokens already written do not change, so they can be kept and reused. Scoring a
new token by running every earlier token through all of GPT-2's 12 layers again
repeats that work. Keeping the keys
and values makes the new ids come out the same, with less arithmetic. This is
called a KV cache, short for key-value cache.

For "The capital of France is" and six new tokens, greedy decoding runs
5 + 6 + 7 + 8 + 9 + 10 = 45 rows through each layer. Keeping the keys and values
runs 10: the 5 prompt rows once, then one row for each new token after the
first. That saving matters when you run GPT-2's real weights later, where an
engine that recomputes the whole sequence at every step can run out of time.

### The `generate` command

The command is the same one as greedy decoding. `tokens.json` again holds a
count, and there is no temperature. The ids must match a run that recomputes the
whole sequence at every step.

A key and a value are the lists of numbers from step 1 of attention, after
`c_attn` and split into heads. Keep them in a list for each layer and each head:

```text
keys[layer][head], values[layer][head] = empty lists

rows = the prompt's embeddings
run the blocks on rows, appending each row's key and value to the lists

repeat count times:
    pick the next id from the last row's logits
    if that was the last new id, stop
    row = the embedding of that id, at its position in the whole sequence
    for each layer:
        run the block on this one row
        // inside attention, for each head:
        append the row's key and value to keys[layer][head], values[layer][head]
        score the row's query against every key in keys[layer][head]
```

The softmax and the output projection are unchanged. The layer norms and the MLP
work on one row at a time, so they only need the new row.

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

- The new token's position is still its index in the whole sequence. The cache
  does not change the position row you add.
- A prompt is cached in order, left to right, before the first new token.
