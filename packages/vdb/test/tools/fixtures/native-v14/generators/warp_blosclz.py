"""Writes a BloscLZ-compressed .nvdb the way NVIDIA Warp does (Volume.save_to_nvdb(codec="blosc")).

Usage: python -I warp_blosclz.py <out-dir>
Needs warp-lang (1.19.0 used) and blosc (python-blosc 1.11.4, c-blosc 1.21.x; blosc.compress
defaults: blosclz, clevel 9, byte shuffle, typesize 8). Writes warp_sphere_blosclz.nvdb and
warp-probes.json (the decoded image's SHA-256 and dense values at a few voxels).
"""

import hashlib
import json
import os
import sys

import numpy as np
import warp as wp

out = sys.argv[1]
wp.init()
n = 12
z, y, x = np.mgrid[0:n, 0:n, 0:n]
d = np.sqrt((x - 5.5) ** 2 + (y - 5.5) ** 2 + (z - 5.5) ** 2)
# load_from_numpy indexes the array [x, y, z]
dense = np.clip(1.0 - d / 6.0, 0, 1).astype(np.float32)
v = wp.Volume.load_from_numpy(dense, min_world=(0, 0, 0), voxel_size=0.5, bg_value=0.0, device="cpu")
path = os.path.join(out, "warp_sphere_blosclz.nvdb")
v.save_to_nvdb(path, codec="blosc")
image = v.array().numpy().tobytes()
probes = []
for ijk in [(0, 0, 0), (5, 5, 5), (6, 5, 4), (11, 0, 3), (2, 9, 7), (11, 11, 11)]:
    probes.append({"ijk": list(ijk), "value": float(dense[ijk[0], ijk[1], ijk[2]])})
with open(os.path.join(out, "warp-probes.json"), "w") as f:
    json.dump(
        {"warp": wp.__version__, "imageBytes": len(image), "imageSha256": hashlib.sha256(image).hexdigest(), "probes": probes},
        f,
        indent=2,
    )
print(path, os.path.getsize(path), len(image))
