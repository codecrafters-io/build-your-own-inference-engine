In this stage, you'll implement the `inspect` command to print what's inside a
trained model's file. Generation needs the weights read correctly first, and
`inspect` is how you check what your engine loaded.

### How an inference engine works

An inference engine is the program that runs a trained model. The model here is
GPT-2, and it comes already trained, as a file of numbers called its weights.

Those numbers sit in a file called `model.safetensors`, in 160 named tables.
The tables are called
[tensors](https://app.codecrafters.io/concepts/tensors). They hold numbers
arranged in different ways, for example 4 numbers in a single list, or 6
numbers in 2 rows of 3.

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
word or a part of a word. For example, the prompt "The capital of France is"
becomes the ids `[464, 3139, 286, 4881, 318]`.

The weights do nothing on their own. Reading the file, running the layers, and
choosing each next token are the engine's work, and the engine is what you
build.

| Step | What it does |
| --- | --- |
| Inspect | Reads every tensor's name, shape, and first numbers |
| Embed | Turns each token id into a row of numbers |
| Attend | Mixes earlier tokens into the current one |
| Score | Scores every possible next token |
| Pick | Takes the highest score, or samples |
| Repeat | Feeds the new token back in, reusing past work |
| Run | Does all of it on GPT-2's real weights |

Generation itself starts at Embed, with the token ids.

`inspect` generates nothing, and it keeps earning its place after this stage.
When numbers come out wrong, it shows what your engine actually loaded.

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
