import numpy as np
import pytest

pytest.importorskip("tensorflow")

from deforestation.model import build_unet


def test_unet_output_shape():
    model = build_unet((64, 64, 5), base=8, depth=2)
    out = model.predict(np.zeros((1, 64, 64, 5), dtype="float32"), verbose=0)
    assert out.shape == (1, 64, 64, 1)
    assert 0.0 <= out.min() and out.max() <= 1.0
