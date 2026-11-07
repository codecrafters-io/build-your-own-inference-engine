In this stage, you'll implement the `inspect` command to print what's inside a
model file.

### How an inference engine works

A trained model like GPT-2 is a file of numbers, called its weights. The model
doesn't run by itself. An inference engine is the program that reads the
weights and does the arithmetic that turns a prompt into the next word. In this
challenge, the weights are given, and you build the engine.

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

Each stage adds one command to your engine:

| Stage | What your engine does |
| --- | --- |
| Read safetensors | Reads the weights, which is this stage |
| Embed tokens | Turns each token id into a row of numbers |
| Causal attention | Mixes each row with the rows before it |
| Transformer block | Runs attention, then reworks each row on its own |
| Logits | Scores every token as the next one |
| Greedy decode | Picks the top-scoring token, again and again |
| Sample a token | Picks a token at random, weighted by its score |
| KV cache | Reuses work from earlier steps |
| Run GPT-2 | Runs it all on GPT-2's real weights |

### Why inspect?

Each table in the file is called a tensor. A tensor is a list of numbers plus a
shape, which says how to read the list. For example, 4 numbers with shape `[4]`
are a single list, and 6 numbers with shape `[2, 3]` are 2 rows of 3. A tensor
also has a `dtype`, short for data type, the kind of number it holds. `F32` is a
32-bit float, and `I32` is a 32-bit integer.

`inspect` prints every tensor's name, `dtype`, shape, and first numbers. When
numbers in later stages come out wrong, it's how you check what you loaded.

### The `inspect` command

`inspect` takes the path of a safetensors file:

```bash
$ ./your_program.sh inspect model.safetensors
```

The starter code already reads the file. Its `load_tensors` function returns
every tensor's `dtype`, shape, and numbers, by name. The
[safetensors format](https://github.com/huggingface/safetensors#format)
describes how the file stores them, if you want to see how `load_tensors` works.

Print a JSON array with one object per tensor, sorted by name. Each object has
the tensor's `name`, `dtype`, and `shape`, and `first`, its first 4 numbers. A
tensor with fewer than 4 numbers prints them all. For a tensor named `scale`
whose four numbers are `1`, `0.5`, `0.25`, and `0.125`, the output is:

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

There's no model arithmetic in this stage. Later stages do the computing, on
the tensors `load_tensors` returns.

### Tests

The tester will write a safetensors file to `model.safetensors` and execute your
program like this:

```bash
$ ./your_program.sh inspect model.safetensors
```

Your program must print a JSON array:

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
It will also run a file whose tensors are not in name order, a file with two
tensors at different offsets, and a file with an `I32` tensor.

### Notes

{{#lang_is_go}}
- In Go, the starter code's loader is called `loadTensors`.
{{/lang_is_go}}
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) walks
  through GPT-2 with diagrams, from tokens to the next word. Read it once for
  the big picture. Each stage explains its own piece again where you build it.
