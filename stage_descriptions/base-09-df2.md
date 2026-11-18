In this stage, you'll add sampling to `generate`, to pick each new token at
random, weighted by its probability.

### Why sample?

With fixed model weights, greedy decoding gives a prompt the same continuation
on each run. Sampling allows different continuations by choosing among tokens
according to their probabilities. A more probable token gets a larger share
of the possible random choices.

Temperature is a positive number that controls how concentrated those
probabilities are. Dividing logits by a temperature below 1 before softmax
favors the highest scores more strongly. A temperature above 1 spreads the
probability more evenly. Here are GPT-2's probabilities for the
first token after "The capital of France is":

| Token | Temperature 1 | Temperature 0.7 |
| --- | --- | --- |
| " the" | 0.085 | 0.227 |
| " now" | 0.048 | 0.101 |
| " a" | 0.046 | 0.096 |
| " France" | 0.032 | 0.058 |
| " Paris" | 0.032 | 0.057 |

Top-p limits the set of tokens available for selection. It keeps the most
likely tokens until their probabilities first reach or exceed `top_p`.
With a `top_p` of 0.9, that's
1,503 tokens at temperature 1, and 62 at temperature 0.7.

### Sampling one token

`tokens.json` gains two fields:

```json
{ "count": 1, "temperature": 1, "top_p": 1 }
```

When `tokens.json` has no `temperature`, keep taking the highest score as
before. The tester still runs those files, with nothing on stdin.

Stdin has one number per new token, each between 0 and 1. Read them as text, one
per line. These supplied numbers make the test repeatable, so don't generate
your own random numbers.

For each prediction, use softmax on `logit / temperature` to get probabilities.
Order tokens by descending probability, with the lower ID first on a tie.
Keep the shortest prefix of that order whose total probability reaches or
exceeds `top_p`. The token that reaches the threshold stays in the set.

Divide the retained probabilities by their total so they sum to 1 again.
In the same order, each token occupies an interval whose width is its new
probability. Choose the first token whose cumulative probability is strictly
greater than the next number from stdin. As with greedy decoding, the selected
ID joins the sequence for the next prediction, and stdout contains only the
new IDs.

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

The ids have to match exactly. The tester will also run two models with random
weights, asking each for four new tokens and sending four random numbers. One
has a `top_p` below 1, and the other has a temperature other than 1.

### Notes

- `top_p` of 1 keeps every token that has a probability above 0.
- Very negative inputs to an exponential can underflow to zero, while very
  positive inputs can overflow. For a stable softmax, subtract the largest
  scaled logit before taking exponentials. This leaves the probabilities
  unchanged because softmax depends only on differences between its inputs.
