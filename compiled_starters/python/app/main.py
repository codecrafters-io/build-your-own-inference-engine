import json
import struct
import sys


def main():
    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    # TODO: Uncomment the code below to pass the first stage
    #
    # command = sys.argv[1]
    # if command == "inspect":
    #     with open(sys.argv[2], "rb") as f:
    #         header_len = struct.unpack("<Q", f.read(8))[0]
    #         header = json.loads(f.read(header_len))
    #         data = f.read()
    #     tensors = []
    #     for name, info in header.items():
    #         if name == "__metadata__":
    #             continue
    #         start, end = info["data_offsets"]
    #         raw = data[start:end]
    #         count = 1
    #         for dim in info["shape"]:
    #             count *= dim
    #         take = min(4, count)
    #         dtype = info["dtype"]
    #         if dtype == "F32":
    #             values = list(struct.unpack("<" + "f" * take, raw[: take * 4]))
    #         elif dtype == "I32":
    #             values = list(struct.unpack("<" + "i" * take, raw[: take * 4]))
    #         else:
    #             print(f"unsupported dtype {dtype}", file=sys.stderr)
    #             raise SystemExit(1)
    #         tensors.append(
    #             {
    #                 "name": name,
    #                 "dtype": dtype,
    #                 "shape": info["shape"],
    #                 "first": values,
    #             }
    #         )
    #     tensors.sort(key=lambda item: item["name"])
    #     print(json.dumps(tensors))
    # else:
    #     print(f"Unknown command {command}", file=sys.stderr)
    #     raise SystemExit(1)


if __name__ == "__main__":
    main()
