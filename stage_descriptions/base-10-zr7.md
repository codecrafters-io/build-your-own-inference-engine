In this stage, you'll run your `generate` command on GPT-2's real weights.

### GPT-2's Real Weights

Every weights file so far was a tiny model, built to check one step. GPT-2's real file uses the same tensor names, at full size: 160 tensors holding 124 million weights. Your engine already runs every step GPT-2 runs, so on the real weights it should write the same words GPT-2 does.

For example, take the prompt "The capital of France is". Its ids are `[464, 3139, 286, 4881, 318]`. GPT-2 continues it with " the capital of the French Republic", which is these six ids:

```json
[262, 3139, 286, 262, 4141, 2066]
```

Here's the size of each part, in GPT-2's real file:

```text
wte.weight          [50257, 768]    a row for each of 50,257 tokens
wpe.weight          [1024, 768]     a row for each of 1,024 positions
12 layers           12 heads each, with 64 numbers per head
MLP                 768 → 3,072 → 768 numbers
```

The file also has an `h.<layer>.attn.bias` tensor in every layer. It's GPT-2's causal mask, which marks the later rows that attention skips, saved with the weights. Your attention already skips later rows, so you can ignore these tensors. They're not the same as `h.<layer>.attn.c_attn.bias`, which you do use.

### Memory and Speed

GPT-2's weights file holds 548 MB of 32-bit floats, and the test servers give your program 2 GB of memory. The starter code's loader keeps every number as a 32-bit float, so keep the numbers the way it returns them.{{#lang_is_python}} In Python, it returns each tensor's numbers as a `memoryview` of the file's bytes, which doesn't copy them. Don't turn whole tensors into lists: a list of Python floats takes about 32 bytes per number, which is about 4 GB for GPT-2's weights.{{/lang_is_python}}

This test also takes longer than the earlier ones, since your engine runs all 12 layers for every new token.{{#lang_is_python}} A Python engine with a KV cache took about 30 seconds on a laptop and about 90 seconds on the test servers. The tester stops your program after 300 seconds, so an engine without a KV cache runs out of time.{{/lang_is_python}}{{^lang_is_python}} A {{language_name}} engine finishes in a few seconds.{{/lang_is_python}}

### Tests

The tester will link GPT-2's weights to `model.safetensors`, then write a `config.json`, an `ids.json`, and a `tokens.json`. Next, it will execute your program like this:

```bash
$ ./your_program.sh generate model.safetensors config.json ids.json tokens.json
```

`config.json` is GPT-2's:

```json
{"n_embd": 768, "n_head": 12, "n_layer": 12, "n_positions": 1024, "vocab_size": 50257}
```

`ids.json` is the prompt "The capital of France is", `[464, 3139, 286, 4881, 318]`, and `tokens.json` is `{"count": 6}`.

The tester will verify that:

- Your program prints only a JSON array to stdout, with 6 ids
- The ids are exactly the ones GPT-2 writes for this prompt
- Your program finishes within 300 seconds

It will also turn your ids back into text and print what your engine wrote. Before the test, the tester downloads GPT-2's weights, which takes a few seconds.

### Notes

- To run GPT-2 on your own machine, download its weights from [Hugging Face](https://huggingface.co/openai-community/gpt2) into a folder outside your repository, so the 548 MB file never ends up in a commit. Your README's "Running GPT-2" section shows how. Then use the `config.json` above.
- To read your ids as text, download [`vocab.json`](https://huggingface.co/openai-community/gpt2/blob/main/vocab.json) from the same page. It maps each token's text to its id, and a leading `Ġ` stands for a space. For a prompt whose words are each a single token, look up each word there to build `ids.json`.
- For this prompt, GPT-2's top two logits get very close after the sixth new token. At the seventh step, they differ by 0.007, so an engine that does its arithmetic in 32-bit floats and one that uses 64-bit floats may pick different tokens. That's why the test stops at six, and it doesn't mean either engine is wrong.{{#lang_is_python}} Python does its arithmetic in 64-bit floats, even though the file stores 32-bit ones.{{/lang_is_python}}
