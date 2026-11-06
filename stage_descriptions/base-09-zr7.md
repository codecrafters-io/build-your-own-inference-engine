In this stage, you'll run your `generate` command on GPT-2's real weights.

### Why run GPT-2?

Every file so far was a tiny model built to check one rule. GPT-2's real file
uses the same tensor names, at full size: 124 million numbers in 548 MB. Your
engine already runs every step GPT-2 runs, so on the real weights it should
write the same words GPT-2 does.

For example, take the prompt "The capital of France is". Its ids are
`[464, 3139, 286, 4881, 318]`. GPT-2 continues it with " the capital of the
French Republic", which is these six ids:

```json
[262, 3139, 286, 262, 4141, 2066]
```

### Loading 548 MB

The tensors are the ones earlier stages read. `wte.weight` has shape
`[50257, 768]`, and there are 12 layers with 12 heads each. The file also has an
`h.<layer>.attn.bias` tensor in every layer. It is GPT-2's causal mask, saved
with the weights, and you can skip it because your attention already drops the
later scores.

Keep each number as a 32-bit float, read from the file's bytes. Storing every
number as a separate object multiplies the memory the weights need, and the test
server gives your program about 2.5 GB.

{{#lang_is_python}}
A Python `float` takes 24 bytes, so a list of GPT-2's numbers needs about 6 GB,
more than the test server allows. A
[`memoryview`](https://docs.python.org/3/builtins/stdtypes.html#memoryview.cast)
reads the same bytes as floats, with no copy:

```python
values = memoryview(data)[start:end].cast("f")
```

`values[i]` is still a Python `float`, but the view stores 4 bytes per number.
{{/lang_is_python}}
{{#lang_is_go}}
Decode each tensor into a `[]float32` with `binary.LittleEndian.Uint32` and
`math.Float32frombits`. That keeps the weights at about 550 MB.
{{/lang_is_go}}
{{#lang_is_rust}}
Decode each tensor into a `Vec<f32>` with `f32::from_le_bytes`. That keeps the
weights at about 550 MB.
{{/lang_is_rust}}

### Tests

The tester will link GPT-2's weights to `model.safetensors`, write
`config.json`, `ids.json`, and `tokens.json`, then execute your program like
this:

```bash
$ ./your_program.sh generate model.safetensors config.json ids.json tokens.json
```

`config.json` is GPT-2's:

```json
{ "n_embd": 768, "n_head": 12, "n_layer": 12, "n_positions": 1024, "vocab_size": 50257 }
```

`ids.json` is `[464, 3139, 286, 4881, 318]`, and `tokens.json` is
`{"count": 6}`. Your program must print:

```json
[262, 3139, 286, 262, 4141, 2066]
```

The ids have to match exactly. The tester will also turn your ids back into
text and print what your engine wrote, " the capital of the French Republic".

This test takes longer than the earlier ones. The tester first downloads GPT-2's
weights, which takes 5 to 15 seconds, and then your engine runs all 12 layers
for every token.
{{#lang_is_python}}
A Python engine that reuses keys and values takes about 45 seconds on a laptop,
and can take longer on the test servers. The tester stops your program after
300 seconds, so an engine that recomputes the whole sequence at every step can
run out of time.
{{/lang_is_python}}
{{^lang_is_python}}
A {{language_name}} engine finishes in a few seconds.
{{/lang_is_python}}

### Notes

- To run GPT-2 on your own machine, download its weights from
  [Hugging Face](https://huggingface.co/openai-community/gpt2) into a folder
  outside your repository, so the 548 MB file never ends up in a commit. Then
  use the `config.json` above:

    ```bash
    mkdir -p /tmp/testing && cd /tmp/testing
    curl -L -o model.safetensors https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
    ```

- To read your ids as text, download
  [`vocab.json`](https://huggingface.co/openai-community/gpt2/blob/main/vocab.json)
  from the same page. It maps each token's text to its id, and a leading `Ġ`
  stands for a space. For a prompt whose words are each a single token, look up
  each word there to build `ids.json`.
- Past six tokens, GPT-2's top two scores can be very close. At the seventh
  step for this prompt they differ by 0.007, so an engine that uses 32-bit
  floats and one that uses 64-bit floats may choose different tokens. That
  doesn't mean either one is wrong.
