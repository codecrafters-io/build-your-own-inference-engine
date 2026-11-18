In this stage, you'll reuse past keys and values in `generate` instead of
recomputing the whole sequence.

### Why keep keys and values?

A new token needs its own query, key, and value. The keys and values of the
tokens already processed do not change when a token is appended. Causal
attention prevents earlier positions from depending on later tokens, so their
computed rows, keys, and values stay valid. Scoring a new token by running
every earlier token through all of GPT-2's 12 layers again repeats that work.
Keeping the keys and values makes the new ids come out the
same, with less arithmetic. This is called a KV cache, short for key-value
cache.

For "The capital of France is" and six new tokens, greedy decoding runs
5 + 6 + 7 + 8 + 9 + 10 = 45 rows through each layer. Keeping the keys and values
needs only 10: the 5 prompt rows once, then 5 new rows. The last selected token
doesn't need a forward pass because no further prediction follows it. That
saving matters when you run GPT-2's real weights later, where an engine that
recomputes the whole sequence at every step can run out of time.

### The `generate` command

The command is the same one as greedy decoding. `tokens.json` again holds a
count, and there is no temperature. The ids must match a run that recomputes the
whole sequence at every step.

A cached key or value is the vector produced by `c_attn` for one position,
split into heads as before. Each layer and head has its own keys and values,
retained in position order. The prompt's keys and values must be available
when processing generated tokens.

For a new row, attention uses that row's query with the cached keys and its own
new key. Its weighted sum uses the corresponding cached values and its own
new value. Earlier rows don't need to pass through the blocks again.

The softmax and output projection are unchanged. Layer normalization and the
MLP work independently on each row, so they only need the new row too. The
cache's storage layout is your choice.

In this example `ids.json` is `[0, 1, 0]` and `tokens.json` is
`{ "count": 3 }`. The model has two tokens, and the rows of `wpe.weight` differ
from each other, so a new token embedded at the wrong position changes what
comes out. The ids are:

```json
[0, 1, 1]
```

### Tests

The tester will write the four files and execute your program like this:

```bash
$ ./your_program.sh generate model.safetensors config.json ids.json tokens.json
```

Your program must print:

```json
[0, 1, 1]
```

The ids have to match exactly. The tester will also run a two-id prompt, where
embedding each new token at position 0 instead of at its index in the whole
sequence changes the ids, and a two-layer model with random weights, where each
layer needs keys and values of its own.

### Notes

- The new token's position is still its index in the whole sequence. The cache
  does not change the position row you add.
- A prompt is cached in order, left to right, before the first new token.
