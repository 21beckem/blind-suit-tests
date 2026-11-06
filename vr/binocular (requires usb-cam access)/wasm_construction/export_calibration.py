"""Export the existing NumPy calibration archive without a NumPy dependency."""

import ast
import json
import struct
import sys
import zipfile
from pathlib import Path


def read_npy(data):
    if data[:6] != b"\x93NUMPY":
        raise ValueError("Invalid NPY header")
    major = data[6]
    header_size, offset = ((struct.unpack_from("<H", data, 8)[0], 10)
                           if major == 1 else (struct.unpack_from("<I", data, 8)[0], 12))
    header = ast.literal_eval(data[offset:offset + header_size].decode("latin1"))
    if header["fortran_order"]:
        raise ValueError("Fortran-order calibration arrays are unsupported")
    dtype = header["descr"]
    formats = {"<f8": "d", "<f4": "f", "<i8": "q", "<i4": "i"}
    if dtype not in formats:
        raise ValueError(f"Unsupported calibration dtype: {dtype}")
    shape = header["shape"]
    count = 1
    for dim in shape:
        count *= dim
    values = list(struct.unpack_from("<" + formats[dtype] * count,
                                     data, offset + header_size))

    def reshape(items, dims):
        if not dims:
            return items[0]
        if len(dims) == 1:
            return items
        stride = len(items) // dims[0]
        return [reshape(items[i * stride:(i + 1) * stride], dims[1:])
                for i in range(dims[0])]

    return reshape(values, shape)


def main(source, target):
    with zipfile.ZipFile(source) as archive:
        result = {Path(name).stem: read_npy(archive.read(name))
                  for name in archive.namelist() if name.endswith(".npy")}
    required = ("K1", "D1", "K2", "D2", "R1", "R2", "P1", "P2", "image_size")
    for key in required:
        if key not in result:
            raise ValueError(f"Calibration is missing {key}")
    Path(target).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {source} to {target}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

