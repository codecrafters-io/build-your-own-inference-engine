In this stage, you'll implement the `generate` command to write new tokens, always picking the one with the highest logit.

### Greedy Decoding

`logits` scores every token as the next one, but writing text means choosing one. The simplest choice is the token with the highest logit. GPT-2 then appends that token to the prompt, scores the longer prompt again, and picks the highest again, one token at a time. This is called greedy decoding. It uses no randomness, so the same prompt always continues the same way.

For example, here's GPT-2 continuing "The capital of France is", one step at a time. The lead is how much higher the picked token's logit is than the runner-up's:

| Step | Picked | Runner-up | Lead |
| --- | --- | --- | --- |
| 1 | " the" | " now" | 0.57 |
| 2 | " capital" | " city" | 1.62 |
| 3 | " of" | "," | 3.87 |
| 4 | " the" | " France" | 0.33 |
| 5 | " French" | " Republic" | 0.96 |
| 6 | " Republic" | " Empire" | 1.32 |

At step 4, a lead of 0.33 is all that separates " the capital of the French Republic" from " the capital of France". The six picked ids are `[262, 3139, 286, 262, 4141, 2066]`.

### The `generate` Command

`generate` takes four arguments:

1. `model.safetensors`, the same weights file as for `logits`
2. `config.json`, the same format as for `logits`
3. `ids.json`, the prompt's token ids
4. `tokens.json`, a JSON object that says how many new tokens to write, like `{"count": 2}`

It prints the new ids as a JSON array, without the prompt's ids. Your program must repeat these steps `count` times:

1. Compute the logits for the ids, the same way `logits` does
2. Pick the id with the highest logit. If several ids tie for the highest, pick the lowest of them
3. Append the picked id to the ids, so the next step's logits take it into account

For example, if a step's logits are `[2, 5, 5, 1]`, ids 1 and 2 tie for the highest, so the step picks id 1.

### Tests

The tester will write a `model.safetensors`, a `config.json`, an `ids.json`, and a `tokens.json`, then execute your program like this:

```bash
$ ./your_program.sh generate model.safetensors config.json ids.json tokens.json
```

The tester will verify that:

- Your program prints only a JSON array to stdout, with `count` ids
- Each id is the one greedy decoding picks at that step

It will also run a file where every token has the same logit, so every step is a tie.
