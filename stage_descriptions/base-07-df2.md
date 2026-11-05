In this stage, you'll add temperature and top-p sampling to `generate`, using
one random number from stdin per token.

### Why sample?

Greedy decoding always appends the same token. Sampling picks a token at random,
weighted by its score, and temperature and top-p shape those weights.
Temperature divides the scores before the softmax, so a temperature below 1
makes the highest score even larger relative to the others. Top-p then throws
away the least likely tokens.

### Sampling one token

`tokens.json` gains two fields:

```json
{ "count": 1, "temperature": 1, "top_p": 1 }
```

When `tokens.json` has no `temperature`, keep taking the highest score as
before. The tester still runs those files, with nothing on stdin.

Stdin has one number per new token, each between 0 and 1. Read them as text, one
per line.

For each new token:

1. Divide every logit by `temperature`.
2. Softmax, so the values are probabilities and add up to 1.
3. Sort the tokens by probability, highest first. Equal probabilities keep the
   lower token id.
4. Keep the shortest prefix of that list whose probabilities add up to at least
   `top_p`.
5. Divide those kept probabilities by their sum, so they add up to 1 again.
6. Walk the list, adding as you go. Choose the first token whose running total
   is greater than the random number for this step.

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

- Read `count` numbers from stdin, in order, one per new token.
- Equal probabilities stay in token-id order. The lower id comes first.
- `top_p` of 1 keeps every token that has a probability above 0. The running
  total uses the renormalized probabilities.
