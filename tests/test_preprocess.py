import numpy as np

from deforestation.preprocess import spatial_split, tile


def test_tile_skips_invalid_patches():
    feats = np.random.rand(512, 512, 5).astype("float32")
    label = np.random.randint(0, 2, (512, 512)).astype("uint8")
    valid = np.ones((512, 512), dtype=bool)
    valid[:256, :256] = False
    X, y, pos = tile(feats, label, valid, size=256, min_valid=0.9)
    assert X.shape == (3, 256, 256, 5) and y.shape == (3, 256, 256, 1)
    assert (0, 0) not in map(tuple, pos)


def test_spatial_split_keeps_blocks_together():
    pos = np.array([(r, c) for r in range(0, 4096, 256) for c in range(0, 4096, 256)])
    split = spatial_split(pos, block=1024, seed=0)
    blocks = {}
    for (r, c), s in zip(pos, split):
        blocks.setdefault((r // 1024, c // 1024), set()).add(int(s))
    assert all(len(v) == 1 for v in blocks.values())   # a block never straddles two sets
    assert {0, 1, 2} <= set(split.tolist())
