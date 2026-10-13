import numpy as np
from rasterio.transform import from_origin

from deforestation.change_detection import NODATA, compute_loss, loss_to_polygons, min_pixels_from_ha


def test_detects_block_and_drops_noise():
    a = np.ones((60, 60), dtype="uint8")
    b = a.copy()
    b[10:30, 10:30] = 0   # real clearing: 20 x 20 px
    b[50, 50] = 0         # single-pixel noise
    loss = compute_loss(a, b, min_pixels=20)
    assert loss[15, 15]
    assert not loss[50, 50]
    assert loss.sum() == 400


def test_nodata_is_ignored():
    a = np.ones((40, 40), dtype="uint8")
    b = np.zeros((40, 40), dtype="uint8")
    b[:20, :] = NODATA
    loss = compute_loss(a, b, min_pixels=10)
    assert not loss[:20].any()
    assert loss[20:].all()


def test_regrowth_is_not_loss():
    a = np.zeros((30, 30), dtype="uint8")
    b = np.ones((30, 30), dtype="uint8")
    assert compute_loss(a, b, min_pixels=1).sum() == 0


def test_polygon_area_in_hectares():
    loss = np.zeros((50, 50), dtype=bool)
    loss[10:30, 10:30] = True  # 400 px * 100 m2 = 4 ha
    gdf = loss_to_polygons(loss, from_origin(0, 500, 10, 10), "EPSG:32643")
    assert len(gdf) == 1
    assert abs(gdf.area_ha.sum() - 4.0) < 1e-6


def test_min_pixels_from_ha():
    assert min_pixels_from_ha(0.5, 10) == 50
