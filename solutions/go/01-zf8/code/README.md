This is a starting point for Go solutions to the
["Build Your Own Inference Engine" Challenge](https://codecrafters.io/challenges/inference-engine).

In this challenge, you'll build a program that loads GPT-2's weights and writes
the next token IDs. Along the way you'll learn about attention, the transformer,
and more.

**Note**: If you're viewing this repo on GitHub, head over to
[codecrafters.io](https://codecrafters.io) to try the challenge.

# Passing the first stage

The entry point for your inference engine is in `app/main.go`. Study and
uncomment the relevant code, and then run the command below to execute the tests
on our servers:

```sh
codecrafters submit
```

That's all!

# Stage 2 & beyond

Note: This section is for stages 2 and beyond.

1. Ensure you have `go (1.26)` installed locally
1. Run `./your_program.sh` to run your inference engine, which is implemented in
   `app/main.go`.
1. Run `codecrafters submit` to submit your solution to CodeCrafters. Test
   output will be streamed to your terminal.

# Testing locally

The `your_program.sh` script reads the files you pass on the command line. For
example:

```sh
mkdir -p /tmp/testing && cd /tmp/testing
/path/to/your/repo/your_program.sh inspect model.safetensors
```

To make this easier to type out, you could add a
[shell alias](https://shapeshed.com/unix-alias/):

```sh
alias myengine=/path/to/your/repo/your_program.sh

mkdir -p /tmp/testing && cd /tmp/testing
myengine inspect model.safetensors
```
