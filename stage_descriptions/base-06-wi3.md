In this stage, you'll implement the `logits` command to score every token as the next one.

### Logits

So far, your engine turns a prompt into rows and runs them through one layer's block. GPT-2 runs the blocks of all 12 layers, one after another, with each block's output feeding the next. After the last block, it turns the last row into a score for every token it knows, 50,257 scores in all. A higher score means a more likely next token. These scores are called [logits](https://developers.google.com/machine-learning/glossary#logits).

For example, here are the five highest logits after "The capital of France is". GPT-2's tokens for words usually include the space before the word, so `" the"` is one token:

| Token | Id | Logit |
| --- | --- | --- |
| " the" | 262 | -100.25 |
| " now" | 783 | -100.82 |
| " a" | 257 | -100.86 |
| " France" | 4881 | -101.21 |
| " Paris" | 6342 | -101.21 |

The logits are all near -100, and that's normal. What decides the next token is how they compare, so " the" leads " now" by 0.57.

### Scoring the Last Row

GPT-2 only scores the last row, the row for "is". In every layer, attention has mixed the earlier rows into it, so by now it holds what the whole prompt says.

GPT-2 layer-normalizes that row one last time, with `ln_f.weight` and `ln_f.bias`. Then it scores each token by the dot product of that row and the token's row of `wte.weight`, the same table `embed` reads. There's no separate table for scoring. A token whose row matches the last row closely gets a high score:

```text
last row after layer 11      768 numbers
layer norm with ln_f         768 numbers
· each row of wte.weight     50,257 logits, one for each token
```

### The `logits` Command

`logits` takes three arguments:

1. `model.safetensors`, a weights file with every tensor `embed` and `block` use, plus `ln_f.weight` and `ln_f.bias`, each with shape `[n_embd]`
2. `config.json`, the same JSON object of sizes as before. This stage also uses its `n_layer` field, the number of layers
3. `ids.json`, a JSON array of token ids, like the one `embed` reads

It prints a JSON array of logits, one for each row of `wte.weight`. Your program must:

1. Embed the ids, the same way `embed` does
2. Run the rows through the block of every layer, from layer 0 to layer `n_layer - 1`, with each block's output feeding the next
3. Layer-normalize the last row with `ln_f.weight` and `ln_f.bias`
4. Compute the dot product of that row and each row of `wte.weight`, in order

For example, say `ln_f.weight` is `[0, 0]` and `ln_f.bias` is `[2, 1]`, so the last row after layer norm is `[2, 1]`, whatever the blocks did. If `wte.weight` has the rows `[1, 0]`, `[0, 1]`, and `[1, 1]`, the logits are:

```text
token 0    [2, 1] · [1, 0] = 2
token 1    [2, 1] · [0, 1] = 1
token 2    [2, 1] · [1, 1] = 3
```

So `logits` prints `[2.0, 1.0, 3.0]`.

### Tests

The tester will write a `model.safetensors`, a `config.json`, and an `ids.json`, then execute your program like this:

```bash
$ ./your_program.sh logits model.safetensors config.json ids.json
```

The tester will verify that:

- Your program prints only a JSON array to stdout, with one number for each row of `wte.weight`
- Each number matches the logit for that token, to within 0.001 or 0.1% of the logit, whichever is larger

The tester reads your output as JSON numbers, so `2` and `2.0` both match. It will run a few different files, including a two-token prompt, where only the last position is scored, and a file with two layers.

### Notes

- Step 4 uses each row of `wte.weight`, one row per token, so it isn't the matrix multiplication from earlier stages, which uses a tensor's columns.
- To see GPT-2's real logits, download its weights as your README's "Running GPT-2" section shows. Then run `logits` with `[464, 3139, 286, 4881, 318]` in `ids.json` and GPT-2's sizes in `config.json`:

    ```json
    {"n_embd": 768, "n_head": 12, "n_layer": 12, "n_positions": 1024, "vocab_size": 50257}
    ```

    Id 262, " the", should come out on top at about -100.25.{{#lang_is_python}} This takes about 15 seconds in Python.{{/lang_is_python}}
