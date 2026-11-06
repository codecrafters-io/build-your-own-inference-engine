In this stage, you'll add temperature and top-p sampling to `generate`, using
one random number from stdin per token.

### Why sample?

Greedy decoding always appends the same token, so a prompt can only ever
continue one way. Sampling picks a token at random, weighted by its probability,
so the same prompt can continue differently on each run.

Temperature divides the logits before the softmax. A temperature below 1 makes
the likely tokens even more likely. Here are GPT-2's probabilities for the
first token after "The capital of France is":

| Token | Temperature 1 | Temperature 0.7 |
| --- | --- | --- |
| " the" | 0.085 | 0.227 |
| " now" | 0.048 | 0.101 |
| " a" | 0.046 | 0.096 |
| " France" | 0.032 | 0.058 |
| " Paris" | 0.032 | 0.057 |

Top-p then throws away the least likely tokens. It keeps the most likely ones
until their probabilities add up to `top_p`. With a `top_p` of 0.9, that's
1,503 tokens at temperature 1, and 62 at temperature 0.7.

### Sampling one token

`tokens.json` gains two fields:

```json
{ "count": 1, "temperature": 1, "top_p": 1 }
```

When `tokens.json` has no `temperature`, keep taking the highest score as
before. The tester still runs those files, with nothing on stdin.

Stdin has one number per new token, each between 0 and 1. Read them as text, one
per line. Pick each new token like this:

```text
p = softmax(each logit / temperature)
order = the token ids sorted by p, highest first, the lower id first on a tie
kept = the shortest start of order whose p adds up to at least top_p
divide each kept p by the sum of the kept p, so they add up to 1 again

r = the next number from stdin
total = 0
for each id in kept, in order:
    total = total + p[id]
    if total > r: pick id, and stop
```

In this example the two tokens have the same score, so the probabilities are 0.5
and 0.5. The sorted order is token 0, then token 1. The running totals are 0.5
and 1. The random number is 0.6, which is past 0.5 and not past 1, so the token
is 1.

```json
[1]
```

### Tests

The tester will write the four files, send the random numbers on stdin, and
execute your program like this:

```bash
$ ./your_program.sh generate model.safetensors config.json ids.json tokens.json
```

Your program must print:

```json
[1]
```

The ids have to match exactly. The tester will also run top-p 0.75 with random
number 0.9, which keeps only the leading token, and temperature 0.1 with the
same random number.

### Notes

- `top_p` of 1 keeps every token that has a probability above 0.
