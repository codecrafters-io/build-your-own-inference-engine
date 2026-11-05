# Base course

The learner's program loads a safetensors file of GPT-2 weights and generates token ids. Text in and text out stays in the tokenizer course.

## What the forward pass copies from GPT-2

Checked against `config.json` and the tensor header of `gpt2/model.safetensors`:

- `wte.weight` is `[50257, 768]`, `wpe.weight` is `[1024, 768]`. Position 0 is the first token.
- 12 layers. `n_head` is 12, so each head has 64 numbers. Scores are divided by 8.
- `c_attn.weight` is `[768, 2304]`. The multiply is `x @ weight + bias`, then a split into query, key, and value.
- A position uses itself and earlier positions. Later scores are dropped before the softmax.
- Each block is layer norm, attention, residual, layer norm, MLP, residual. The MLP widens 768 to 3072.
- Layer norm divides by the count of numbers, not one less, and uses epsilon `1e-5`.
- GELU is the tanh approximation, the config's `gelu_new`: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
- Logits are the last position dotted with every row of `wte.weight`. There is no separate output table.
- Greedy ties keep the lower token id.

## Sampling

Temperature, then softmax, then sort by probability with ties broken by the lower token id. Keep the shortest prefix whose mass reaches top-p, renormalize, and take the first token whose running total exceeds the stdin number. This is the course's sampler. It is not Hugging Face's sampler. Top-k was dropped. One uniform number arrives on stdin per new token.

## What the tester runs today

Stages 2 through 8 run on hand-sized files so the first case of each stage can be checked by hand.

Stage 9 (`zr7`) runs the learner's `generate` on the published 124M file. The tester reads it from `CODECRAFTERS_GPT2_WEIGHTS_PATH`, or downloads it from a public URL in the test runners' region to `$TMPDIR/codecrafters-gpt2/` on the container's disk, streaming rather than holding it in memory, and keeps it only if its size and SHA-256 match. It then links the file into the working directory. A later run in the same pod reuses the file, and a new pod downloads it again in 5 to 15 seconds. No other stage downloads it.

- The prompt is "The capital of France is", ids `[464, 3139, 286, 4881, 318]`. GPT-2's greedy continuation is " the capital of the French Republic", ids `[262, 3139, 286, 262, 4141, 2066]`, and the tester requires those six exactly.
- The stage stops at six tokens because each of those steps has a gap of at least 0.33 between the top two logits. The seventh step's gap is 0.007, small enough for a float32 engine and a float64 engine to disagree.
- The program gets 300 seconds, and the stage 6 minutes, which covers the download and stays under the platform's 10-minute limit. The Python reference, which reads weights through `memoryview(...).cast("f")`, takes about 42 seconds on an Apple Silicon laptop and uses 1.1 GB. Unpacking every weight into a Python `float` uses 6.2 GB, which is why the stage tells Python learners how to store them.
- The tester decodes the learner's ids with GPT-2's `vocab.json` and prints the text, on a pass and on a failure.

Learners run GPT-2 at home with the download steps in that stage's Notes and in the starter README.
