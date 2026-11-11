In this stage, you'll make `generate` keep each token's keys and values, instead of recomputing the whole sequence for every new token.

### The KV Cache

So far, `generate` runs the whole sequence through every layer for each new token. But attention only lets a row look back, so adding a new token never changes the keys and values of the tokens before it. They can be kept and reused. Only the new token needs its own query, key, and value. This is called a [KV cache](https://en.wikipedia.org/wiki/Attention_(machine_learning)#KV_cache), short for key-value cache.

For example, for "The capital of France is" and six new tokens, recomputing runs 5 + 6 + 7 + 8 + 9 + 10 = 45 rows through each layer. With a KV cache, it runs only 10:

```text
                             rows run through each layer   keys in each layer's cache
the prompt                   5                             5
after picking " the"         1                             6
after picking " capital"     1                             7
after picking " of"          1                             8
after picking " the"         1                             9
after picking " French"      1                             10
after picking " Republic"    none, it's the last token     10
```

The last new token is never fed back in, so it doesn't need a row.

That saving matters when you run GPT-2's real weights in later stages.{{#lang_is_python}} There, a Python engine took about 90 seconds on the test servers with a KV cache, and ran past their 300-second limit without one.{{/lang_is_python}}

### Running One Row

With a KV cache, each layer keeps a list of the keys and a list of the values it has computed so far. To add a new token, your engine runs just its row through each layer's block:

- In attention, the row's key and value are appended to the layer's lists. Then the row's query is scored against every key in the list, with each head using its own slices as before.
- Layer norm and the MLP work on one row at a time anyway, so they only need the new row.

The new token's row is its embedding at its position in the whole sequence. The cache doesn't change which row of `wpe.weight` you add.

### The `generate` Command

The command and its files are the same as before. The ids it prints must be the same as without a cache. Your program must:

1. Run all of the prompt's rows through every layer's block, from layer 0 to the last, appending each row's key and value to that layer's lists
2. Pick the next id from the logits of the last row
3. If that wasn't the last new id, embed it at its position in the whole sequence, run just that row through the blocks of every layer, and go back to step 2

### Tests

The tester will write a `model.safetensors`, a `config.json`, an `ids.json`, and a `tokens.json`, then execute your program like this:

```bash
$ ./your_program.sh generate model.safetensors config.json ids.json tokens.json
```

The tester will verify that:

- Your program prints only a JSON array to stdout, with `count` ids
- Each id is the one greedy decoding picks at that step

It will run a few different prompts. Their files have different rows in `wpe.weight` for each position, so embedding a new token at the wrong position changes the ids.

### Notes

- The tester can only see your ids, so a program without a KV cache passes this stage too.{{#lang_is_python}} You'll need the cache when your engine runs GPT-2's real weights, since a Python engine without one runs out of time.{{/lang_is_python}}{{^lang_is_python}} The cache makes your engine several times faster when it runs GPT-2's real weights.{{/lang_is_python}}
- A layer's cache holds the keys and values of every head. If you keep them as full rows, take each head's slices out of them, the same way as for the query.
