In this stage, you'll build `inspect`, the first command of your inference
engine. It opens a trained model's file and prints what's inside.

### How an inference engine works

An inference engine is the program that runs a trained model. The model here is
GPT-2, and it comes already trained, as a file of numbers called its weights.

GPT-2's weights are in a file called `model.safetensors`. It holds 160 named
tables of numbers:

```text
model.safetensors
├── wte.weight      one row of 768 numbers for each of 50,257 tokens
├── wpe.weight      one row of 768 numbers for each of 1,024 positions
├── h.0.…           13 tables for layer 0
├── h.1.…           13 tables for layer 1, and so on up to h.11
├── ln_f.weight
└── ln_f.bias
```

GPT-2 doesn't read words. It reads token ids, numbers that each stand for a
word or a part of a word. For example, "The capital of France is" is the ids
`[464, 3139, 286, 4881, 318]`.

The weights do nothing on their own. Reading the file, running the layers, and
choosing each next word is the engine's job, and the engine is what you build.

| Step | What it does | Stages |
| --- | --- | --- |
| Inspect | Reads every tensor's name, shape, and first numbers. Preparation, not generation | Read safetensors |
| Embed | Turns each token id into a row of numbers | Embed tokens |
| Attend | Mixes earlier tokens into the current one | Causal attention, Transformer block |
| Score | Scores every possible next token | Logits |
| Pick | Takes the highest score, or samples | Greedy decode, Sample a token |
| Repeat | Feeds the new token back in, reusing past work | KV cache |
| Run | All of it on GPT-2's real 548 MB file | Run GPT-2 |

You build each row in one or two stages, and the last stage runs them all.
Generation itself starts at Embed, with the token ids. Inspect comes first
because it makes the weights readable, and checkable.

### Why start with inspect?

The model file is a set of named tables of numbers. The tables are called
tensors. They hold numbers arranged in different ways, for example 4 numbers in
a single list, or 6 numbers in 2 rows of 3.

To generate text, your engine does arithmetic with these numbers, so it has to
read them correctly first. `inspect` prints every tensor's name, data type,
shape, and first numbers, so you can see exactly what your engine loaded. It
keeps earning its place after this stage. When numbers in later stages come out
wrong, `inspect` is how you check what you loaded.

### The `inspect` command

`inspect` answers one question: what is in this model file? You point it at a
file, and it prints what it finds:

```bash
$ ./your_program.sh inspect model.safetensors
```

The output is a JSON array with one entry per tensor, sorted by name. Each
entry has the tensor's:

- `name`
- `dtype`, the kind of number it holds, such as `F32`, a 32-bit float
- `shape`, the tensor's sizes. `[4]` is a single list of four numbers.
- `first`, its first 4 numbers, or all of them when the tensor has fewer

For a tensor named `scale` whose four numbers are `1`, `0.5`, `0.25`, and
`0.125`, the output is:

```json
[
  {
    "name": "scale",
    "dtype": "F32",
    "shape": [4],
    "first": [1.0, 0.5, 0.25, 0.125]
  }
]
```

The file uses a format called
[safetensors](https://github.com/huggingface/safetensors#format). The starter
code's loader already reads it, and returns every tensor's `dtype`, shape, and
numbers, by name.

That's the entire exercise: load the tensors, pick out a few values, and print
them as JSON. There's no model arithmetic in this stage. Later stages do the
computing, on the tensors the loader returns.

### Tests

The tester will write a `model.safetensors` file holding the `scale` tensor
from the example, then execute your program like this:

```bash
$ ./your_program.sh inspect model.safetensors
```

Your program must print the JSON from the example above. The tester reads your
output as JSON, so `[1,0.5]` and `[1.0, 0.5]` both pass. It will also run a
file whose tensors are not in name order, a file with two tensors at different
offsets, and a file with an `I32` tensor.
