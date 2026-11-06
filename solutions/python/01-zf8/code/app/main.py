import json
import struct
import sys


def load_tensors(path):
    """Reads every tensor in a safetensors file, with all of its numbers."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()

    formats = {"F32": "f", "I32": "i"}
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        if dtype not in formats:
            raise ValueError(f"unsupported dtype {dtype}")
        start, end = info["data_offsets"]
        # A view of the file's bytes as numbers. It doesn't copy them, so even
        # GPT-2's 548 MB of weights fit in memory.
        values = memoryview(data)[start:end].cast(formats[dtype])
        tensors[name] = {"dtype": dtype, "shape": info["shape"], "values": values}
    return tensors


def main():
    command = sys.argv[1]
    if command == "inspect":
        tensors = load_tensors(sys.argv[2])
        summaries = []
        for name in sorted(tensors):
            tensor = tensors[name]
            summaries.append(
                {
                    "name": name,
                    "dtype": tensor["dtype"],
                    "shape": tensor["shape"],
                    "first": list(tensor["values"][:4]),
                }
            )
        print(json.dumps(summaries))
    else:
        print(f"Unknown command {command}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
