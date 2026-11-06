In this stage, you'll implement the `inspect` command to read a safetensors
file's tensor names, shapes, and values.

### Why inspect?

A model is a set of named tables of numbers, stored in a
[safetensors](https://huggingface.co/docs/safetensors/index) file. Each table is
called a tensor. It has a name, a shape, and a `dtype` (short for data type).

`inspect` prints every table's name, `dtype`, shape, and first values. Later
stages add and multiply these numbers, so you read the file here before you use
it.

### The `inspect` command

The safetensors file starts with an 8-byte
[little-endian](https://en.wikipedia.org/wiki/Endianness#Little-endian) integer.
That integer is the number of header bytes that follow. The header is a JSON
object. Every key is a tensor name, except `__metadata__`, which you skip. A
tensor looks like this:

```json
{ "dtype": "F32", "shape": [4], "data_offsets": [0, 16] }
```

A shape of `[4]` is four numbers. `data_offsets` `[0, 16]` is their byte range
after the header.

Read the first 4 values in that range, or fewer when the tensor is shorter.
`F32` stores each number as a little-endian 32-bit float in 4 bytes, and `I32`
stores a little-endian 32-bit integer.

Print a JSON array sorted by name. For a tensor named `scale` whose four values
are `1`, `0.5`, `0.25`, and `0.125`, the output is:

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

- The first file's header is 90 bytes, which is not a multiple of 8. Use the
  length you read.
