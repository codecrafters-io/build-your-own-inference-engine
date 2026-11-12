In this stage, you'll add sampling to `generate`, to pick each new token at random, weighted by its probability.

### Sampling

Greedy decoding always picks the same token for the same ids, so a prompt can only ever continue one way. Sampling picks the next token at random instead, weighted by its probability, so the same prompt can continue differently each time. The probabilities come from the logits through [softmax](https://nipunbatra.github.io/interactive-articles/softmax-temperature/), the same function attention uses.

Two settings change the probabilities before the pick:

- Temperature makes the likely tokens more or less likely
- Top-p drops the least likely tokens

### Temperature

[Temperature](https://developers.google.com/machine-learning/glossary#temperature) divides every logit before the softmax. Dividing by a temperature below 1 spreads the logits further apart. For example, " the" leads " now" by 0.57, and at a temperature of 0.7 it leads by 0.57 / 0.7 = 0.81. Softmax only depends on these gaps, so the likely tokens become even more likely. A temperature above 1 does the opposite and evens them out.

Here are GPT-2's probabilities for the first token after "The capital of France is":

| Token | Temperature 1 | Temperature 0.7 |
| --- | --- | --- |
| " the" | 0.085 | 0.227 |
| " now" | 0.048 | 0.101 |
| " a" | 0.046 | 0.096 |
| " France" | 0.032 | 0.058 |
| " Paris" | 0.032 | 0.057 |

### Top-p

[Top-p](https://en.wikipedia.org/wiki/Top-p_sampling) keeps only the most likely tokens. It goes through them from the highest probability down, and stops as soon as their probabilities add up to at least `top_p`. The rest can't be picked. For example, with a `top_p` of 0.9, GPT-2 keeps 1,503 tokens at temperature 1, and only 62 at temperature 0.7.

The kept probabilities are then divided by their total, so they add up to 1 again.

### Picking a Token

To pick a token, your program gets a random number between 0 and 1. It walks through the kept tokens, highest probability first, adding up their probabilities as it goes. It picks the first token where the running total is greater than the random number.

For example, say there are three tokens with the logits `[1.099, 1.792, 0]`, the temperature is 1, `top_p` is 0.8, and the random number is 0.95. Since `e^1.099` is 3, `e^1.792` is 6, and `e^0` is 1, the softmax is `3 / 10`, `6 / 10`, and `1 / 10`:

```text
softmax             id 0: 0.3     id 1: 0.6     id 2: 0.1
highest first       id 1 (0.6), id 0 (0.3), id 2 (0.1)
top_p 0.8           keep id 1 (total 0.6), then id 0 (total 0.9), and stop
divide by 0.9       id 1: 0.667   id 0: 0.333
running total       0.667, then 1.0
pick                1.0 is the first total greater than 0.95, so pick id 0
```

### Sampling in `generate`

`tokens.json` can now hold two more fields, like `{"count": 1, "temperature": 1, "top_p": 1}`. When `tokens.json` has both of these fields, pick each new token by sampling instead of taking the highest logit. When it doesn't, keep using greedy decoding as before.

Your program reads the random numbers from stdin, one per line, one for each new token, in order. For each new token, your program must:

1. Compute the logits, the same way `logits` does, and divide each one by `temperature`
2. Softmax them into probabilities
3. Order the ids by probability, highest first, with the lower id first on a tie
4. Keep the ids from the start of that order until their probabilities add up to at least `top_p`, and divide each kept probability by their total
5. Read the next random number from stdin
6. Walk the kept ids in order, adding up their probabilities, and pick the first id where the running total is greater than the random number. If rounding leaves every total at or below it, pick the last kept id

### Tests

The tester will write the four files, send the random numbers on stdin, then execute your program like this:

```bash
$ ./your_program.sh generate model.safetensors config.json ids.json tokens.json
```

The tester will verify that:

- Your program prints only a JSON array to stdout, with `count` ids
- Each id is the one sampling picks with the tester's random numbers

It will run a few different files, including two tokens with the same probability, a `top_p` below 1, and a low temperature. It will also run the greedy files from earlier stages, with nothing on stdin.

### Notes

- Subtract the largest logit from every logit before raising `e` to them. The probabilities stay the same, but the numbers stay in range. GPT-2's logits are all below -100, and `e` to a large negative power rounds to 0: below about -104 in 32-bit floats, and below about -745 in 64-bit floats. At a temperature of 0.1, GPT-2's top logit becomes -1002.5.
- With a `top_p` of 1, top-p keeps every token, so it changes nothing.
