Your engine needs to read the model's numbers correctly before it can make
predictions. In this stage, you'll implement the `inspect` command to report
what's inside a trained model's file.

### How an inference engine works

An inference engine runs a trained model to make predictions. GPT-2 predicts
the next piece of text using numbers called weights, learned from example
text during training. Your engine uses the supplied weights without changing
them.

The model's input is a sequence of token IDs. A token is a piece of text, such
as a word, part of a word, or punctuation, and its ID is an integer label.
For example, "The capital of France is" becomes `[464, 3139, 286, 4881, 318]`.
This challenge supplies IDs as input and expects IDs as generated output.
Converting between text and IDs is a tokenizer's job and is outside this build.

GPT-2's weights are stored in `model.safetensors`. The file holds 160 named
arrays called [tensors](https://app.codecrafters.io/concepts/tensors). A tensor
can be a list of numbers or a table with rows and columns. GPT-2 groups its
calculations into layers, each with its own tensors:

```text
model.safetensors
├── wte.weight      one row of 768 numbers for each of 50,257 tokens
├── wpe.weight      one row of 768 numbers for each of 1,024 positions
├── h.0.…           13 tensors for layer 0
├── h.1.…           13 tensors for layer 1, and so on up to h.11
├── ln_f.weight
└── ln_f.bias
```

Your engine supplies the calculations that use those weights. Its parts fit
together like this:

| Part | Role |
| --- | --- |
| Inspect | Reports loaded tensors so you can check them before generating anything |
| Embed | Turns each token ID into a row of numbers |
| Attend | Combines information from the current and earlier positions |
| Score | Produces a score for every possible next token |
| Pick | Takes the highest score, or samples from probabilities |
| Repeat | Includes the selected token in the next prediction, reusing past work where possible |

The same calculations will eventually run on GPT-2's real 548 MB file. For
now, `inspect` works with small files so you can check their contents by eye.

### Why start with inspect?

`inspect` shows each tensor's name, data type, shape, and first numbers. When
later calculations produce unexpected results, this helps you check whether
the problem started with the loaded data.

### The `inspect` command

Pass the model file's path to `inspect`:

```bash
$ ./your_program.sh inspect model.safetensors
```

Print one JSON array to stdout with one entry per tensor, sorted by name. Each
entry has the tensor's:

- `name`
- `dtype`, the kind of number it holds. `F32` means a 32-bit floating-point
  number, and `I32` means a 32-bit signed integer.
- `shape`, the tensor's sizes. `[4]` is a list of four numbers, and `[2, 3]`
  means two rows of three.
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

File parsing is supplied. There's no model arithmetic in this stage. Later
stages compute with the tensors the loader returns.

### Tests

The tester will write a `model.safetensors` file holding the `scale` tensor
from the example, then execute your program like this:

```bash
$ ./your_program.sh inspect model.safetensors
```

Your program must print:

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

The tester reads your output as JSON, so `[1,0.5]` and `[1.0, 0.5]` both pass.
It will also run a tensor with more than four numbers, a file whose tensors are
not in name order, a file with two tensors, and a file with an `I32` tensor.

### Notes

- Keep stdout for the JSON result and send debug output to stderr.
- Keep the full tensors in the storage the loader supplies. Only the small
  output preview needs to be converted to a JSON-serializable sequence.
