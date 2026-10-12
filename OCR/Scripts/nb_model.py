"""Read Paddle Lite v2 FlatBuffer graphs and parameters without invoking Paddle kernels.

The field layouts follow Paddle-Lite v2.14-rc framework.fbs and param.fbs.
https://github.com/PaddlePaddle/Paddle-Lite/tree/v2.14-rc/lite/model_parser/flatbuffers
"""

import collections
import hashlib
import json
import pathlib
import struct


class FlatBuffer:
    def __init__(self, data):
        self.data = data
        self.root = self.read("I", 0)

    def read(self, kind, offset):
        if offset < 0 or offset + struct.calcsize("<" + kind) > len(self.data):
            raise ValueError("FlatBuffer offset is outside the model.")
        return struct.unpack_from("<" + kind, self.data, offset)[0]

    def field(self, table, index):
        vtable = table - self.read("i", table)
        entry = 4 + index * 2
        if entry >= self.read("H", vtable):
            return None
        relative = self.read("H", vtable + entry)
        return table + relative if relative else None

    def reference(self, position):
        return position + self.read("I", position)

    def scalar(self, table, index, kind="i", default=0):
        position = self.field(table, index)
        return default if position is None else self.read(kind, position)

    def string_at(self, position):
        start = self.reference(position)
        count = self.read("I", start)
        return bytes(self.data[start + 4:start + 4 + count]).decode("utf-8")

    def string(self, table, index):
        position = self.field(table, index)
        return "" if position is None else self.string_at(position)

    def table(self, table, index):
        position = self.field(table, index)
        return None if position is None else self.reference(position)

    def vector(self, table, index, kind="i"):
        position = self.field(table, index)
        if position is None:
            return []
        start = self.reference(position)
        count = self.read("I", start)
        if kind in ("table", "string"):
            positions = [start + 4 + i * 4 for i in range(count)]
            return [self.reference(p) if kind == "table" else self.string_at(p) for p in positions]
        step = struct.calcsize("<" + kind)
        return [self.read(kind, start + 4 + i * step) for i in range(count)]


def read_attr(buffer, table):
    kind = buffer.scalar(table, 1)
    if kind == 0: return buffer.scalar(table, 2)
    if kind == 1: return buffer.scalar(table, 3, "f")
    if kind == 2: return buffer.string(table, 4)
    if kind == 3: return buffer.vector(table, 5)
    if kind == 4: return buffer.vector(table, 6, "f")
    if kind == 5: return buffer.vector(table, 7, "string")
    if kind == 6: return bool(buffer.scalar(table, 8, "B"))
    if kind == 7: return buffer.vector(table, 9, "B")
    if kind == 9: return buffer.scalar(table, 11, "q")
    if kind == 11: return buffer.vector(table, 13, "q")
    if kind == 15: return buffer.scalar(table, 14, "d")
    if kind == 12: return buffer.vector(table, 15, "d")
    raise ValueError(f"Unsupported attribute type: {kind}")


def load(path):
    data = pathlib.Path(path).read_bytes()
    version = struct.unpack_from("<H", data)[0]
    if version != 2:
        raise ValueError(f"Only Paddle Lite model version 2 is supported, got {version}.")
    topology_size = struct.unpack_from("<Q", data, 18)[0]
    topology = FlatBuffer(data[26:26 + topology_size])
    blocks = topology.vector(topology.root, 0, "table")
    if len(blocks) != 1:
        raise ValueError("Expected a single inference block.")
    block = blocks[0]
    variables = {}
    for variable in topology.vector(block, 2, "table"):
        name = topology.string(variable, 0)
        descriptor = topology.table(variable, 1)
        tensor = topology.table(descriptor, 2)
        detail = topology.table(tensor, 0) if tensor is not None else None
        variables[name] = {"shape": topology.vector(detail, 1, "q") if detail is not None else [],
                           "persistable": bool(topology.scalar(variable, 2, "B"))}
    operations = []
    for operation in topology.vector(block, 3, "table"):
        item = {"type": topology.string(operation, 0)}
        for key, field in [("inputs", 1), ("outputs", 2)]:
            item[key] = {topology.string(v, 0): topology.vector(v, 1, "string")
                         for v in topology.vector(operation, field, "table")}
        item["attrs"] = {topology.string(a, 0): read_attr(topology, a)
                         for a in topology.vector(operation, 3, "table")}
        operations.append(item)
    position = 26 + topology_size
    parameter_version, reserved = struct.unpack_from("<HH", data, position)
    if parameter_version != 0:
        raise ValueError("Unsupported parameter version.")
    position += 4 + reserved
    header_size = struct.unpack_from("<H", data, position)[0]
    count = struct.unpack_from("<H", data, position + 2)[0]
    position += 2 + header_size
    parameters = {}
    for _ in range(count):
        total, offset = struct.unpack_from("<II", data, position)
        start = position + 4 + offset
        param = FlatBuffer(data[start:position + 4 + total])
        name = param.string(param.root, 1)
        tensor = param.table(param.root, 3)
        field = param.field(tensor, 4)
        vector = param.reference(field)
        length = param.read("I", vector)
        raw = bytes(param.data[vector + 4:vector + 4 + length])
        parameters[name] = {"shape": param.vector(tensor, 2, "q"),
                            "dtype": param.scalar(tensor, 3), "data": raw}
        position += 4 + total
    if position != len(data):
        raise ValueError("Unparsed bytes remain in the parameter section.")
    return {"version": version, "optimizer": data[2:18].split(b"\0", 1)[0].decode(),
            "sha256": hashlib.sha256(data).hexdigest(), "variables": variables,
            "operations": operations, "parameters": parameters}


def summary(model):
    return {"version": model["version"], "optimizer": model["optimizer"], "sha256": model["sha256"],
            "operations": dict(collections.Counter(op["type"] for op in model["operations"])),
            "parameter_count": len(model["parameters"]),
            "parameter_bytes": sum(len(p["data"]) for p in model["parameters"].values()),
            "parameter_dtypes": dict(collections.Counter(p["dtype"] for p in model["parameters"].values())),
            "inputs": [op for op in model["operations"] if op["type"] == "feed"],
            "outputs": [op for op in model["operations"] if op["type"] == "fetch"]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    args = parser.parse_args()
    print(json.dumps(summary(load(args.model)), indent=2))
