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

Stages 2 through 8 run on hand-sized files so the first case of each stage can be checked by hand. The published 124M file is cached locally and is not yet a tester case. The catalog sentence that quotes a real greedy continuation is still waiting on that run.
