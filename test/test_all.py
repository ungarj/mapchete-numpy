#!/usr/bin/env python
"""Test NumPy read/write driver."""

import os
import sys
import shutil
from copy import copy
import numpy as np

import mapchete
from mapchete.formats import available_output_formats


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "testdata/tmp")


def test_install():
    """Mapchete NumPy driver is installed."""
    assert "NumPy" in available_output_formats()


def test_read_write_single_file():
    """Read/write single file stack."""
    try:
        testtile_id = (5, 1, 1)
        # shape including pixelbuffer of 10
        testdata = np.ones((10, 3, 276, 276))
        with mapchete.open(
            os.path.join(SCRIPT_DIR, "testdata/write_singlefile.mapchete")
        ) as mp:
            tile = mp.config.process_pyramid.tile(*testtile_id)
            tile.data = copy(testdata)
            # write data
            mp.write(tile)
            # read
            with mp.config.output.open(tile, mp) as src:
                assert not src.is_empty()
                output = src.read()
                assert output.ndim == 4
                assert not output.mask.all()
                assert output.shape == (10, 3, 256, 256)
                assert np.where(output == 1, True, False).all()
            # delete
            os.remove(mp.config.output.get_path(tile))
            # open again to clean cache
            with mp.config.output.open(tile, mp) as src:
                # check if empty
                assert src.is_empty()
                # read again
                output = src.read()
                assert output.ndim == 4
                assert output.mask.all()
    finally:
        try:
            shutil.rmtree(OUT_DIR)
        except OSError:
            pass

def test_read_empty():
    """Read empty data."""
    testtile_id = (5, 1, 1)
    with mapchete.open(
        os.path.join(SCRIPT_DIR, "testdata/write_singlefile.mapchete")
    ) as mp:
        tile = mp.config.output_pyramid.tile(*testtile_id)
        # read
        with mp.config.output.open(tile, mp) as src:
            assert src.is_empty()
            output = src.read()
            assert output.ndim == 4
            assert output.mask.all()
            assert output.shape == (1, 3, 256, 256)
            assert np.where(output == 0, True, False).all()

if __name__ == "__main__":
    main(sys.argv)
