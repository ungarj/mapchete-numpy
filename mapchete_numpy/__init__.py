"""Handles output pyramids using GeoTIFFS."""

import os
import numpy as np
import numpy.ma as ma
import bloscpack as bp
import multiprocessing as mp
from functools import partial

from mapchete.formats import base
from mapchete.tile import BufferedTile
from mapchete.io import raster


class OutputData(base.OutputData):
    """Main output class."""

    METADATA = {
        "driver_name": "NumPy",
        "data_type": "raster",
        "mode": "w"
    }

    def __init__(self, output_params):
        """Initialize."""
        super(OutputData, self).__init__(output_params)
        self.path = output_params["path"]
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.file_extension = ".np"
        self.output_params = output_params
        try:
            self.nodata = output_params["nodata"]
        except KeyError:
            self.nodata = 0

    def write(self, process_tile):
        """Write process output into GeoTIFFs."""
        process_tile.data = process_tile.data[:]
        self.verify_data(process_tile.data)
        process_tile.data = self.prepare_data(process_tile.data)
        assert isinstance(process_tile.data, np.ndarray)
        if np.where(process_tile.data == self.nodata, True, False).all():
            return
        for tile in self.pyramid.intersecting(process_tile):
            out_tile = BufferedTile(tile, self.pixelbuffer)
            out_data = raster.extract_from_tile(process_tile, out_tile)
            p = mp.Process(target=self._write_tile, args=(tile, out_data))
            p.start()

    def tiles_exist(self, process_tile):
        """Check whether all output tiles of a process tile exist."""
        return all(
            os.path.exists(self.get_path(tile))
            for tile in self.pyramid.intersecting(process_tile)
        )

    def is_valid_with_config(self, config):
        """Check if output format is valid with other process parameters."""
        assert isinstance(config, dict)
        assert "bands" in config
        assert isinstance(config["bands"], int)
        assert "path" in config
        assert isinstance(config["path"], str)
        assert "dtype" in config
        assert isinstance(config["dtype"], str)
        assert "ndim" in config
        assert isinstance(config["ndim"], int)
        return True

    def get_path(self, tile):
        """Determine target file path."""
        zoomdir = os.path.join(self.path, str(tile.zoom))
        rowdir = os.path.join(zoomdir, str(tile.row))
        return os.path.join(rowdir, str(tile.col) + self.file_extension)

    def prepare_path(self, tile):
        """Create directory and subdirectory if necessary."""
        zoomdir = os.path.join(self.path, str(tile.zoom))
        if not os.path.exists(zoomdir):
            os.makedirs(zoomdir)
        rowdir = os.path.join(zoomdir, str(tile.row))
        if not os.path.exists(rowdir):
            os.makedirs(rowdir)

    def verify_data(self, tile):
        """Verify array data and move array into tuple if necessary."""
        try:
            assert isinstance(tile.data, (np.ndarray, ma.MaskedArray))
        except AssertionError:
            raise ValueError(
                "process output must be a NumPy ndarray or MaskedArray."
                )
        try:
            target_dim = self.output_params["ndim"]
            assert tile.data.ndim in [target_dim, target_dim-1]
        except:
            raise ValueError(
                "process output must have %s dimensions" %
                self.output_params["ndim"]
            )

    def prepare_data(self, data):
        """
        Convert data into correct output.

        Returns a nD masked NumPy array including all bands with the data type
        specified in the configuration.
        """
        if isinstance(data, ma.masked_array):
            masked_data = data
        elif isinstance(data, np.ndarray):
            masked_data = ma.masked_where(data == self.nodata, data, copy=True)
        masked_data.set_fill_value(self.nodata)
        return masked_data.filled()

    def empty(self, process_tile):
        """Empty data."""
        return ma.masked_array(
            data=np.array(
                np.full(
                    (self.output_params["bands"], ) + process_tile.shape,
                    self.output_params["nodata"],
                    dtype=self.output_params["dtype"]),
                ndmin=self.output_params["ndim"]
            ),
            mask=True
        )

    def add_to_stack(
        self, new_array, output_tile, nodata=None, close_gaps=False
    ):
        """Add new array to existing stack."""
        assert isinstance(new_array, np.ndarray)
        stack = self.open(output_tile).read(masked=False)
        if not isinstance(stack, np.ndarray):
            stack = self.empty(output_tile)
        assert isinstance(stack, np.ndarray)
        if not nodata:
            nodata = 0
        if np.where(new_array == nodata, True, False).all():
            return stack
        # Create a new stack putting the new slice on top and a second stack
        # appending an empty slice to the bottom.
        stack1 = np.concatenate((new_array[np.newaxis, :], stack))
        if close_gaps:
            stack2 = np.concatenate((
                stack, np.full(
                    new_array.shape, nodata, dtype=stack.dtype)[np.newaxis, :]))
            if isinstance(new_array.mask, np.bool_):
                if not new_array.mask:
                    return stack1
            mask_stack = np.stack(
                (new_array.mask for i in range(stack1.shape[0])))
            new_stack = np.where(mask_stack, stack2, stack1)
            masked_stack = ma.masked_where(new_stack == nodata, new_stack)
            if isinstance(masked_stack.mask, np.bool_):
                if not masked_stack.mask:
                    return masked_stack
            if masked_stack[-1].mask.all():
                return masked_stack[:-1]
            else:
                return masked_stack
        else:
            return stack1

    def open(self, output_tile, *kwargs):
        """Open process output as input for other process."""
        return InputTile(
            output_tile, self.get_path(output_tile), nodata=self.nodata)

    def _write_tile(self, tile, out_data):
        if isinstance(out_data, np.ndarray) and (
            np.where(out_data == self.nodata, True, False).all()
        ):
            return
        out_path = self.get_path(tile)
        out_tile = BufferedTile(tile, self.pixelbuffer)
        stack = self.add_to_stack(out_data, out_tile)
        self.prepare_path(tile)
        bp.pack_ndarray_file(stack, out_path)


class InputTile(base.InputTile):
    """Target Tile representation of output data."""

    def __init__(self, tile, path, nodata=0):
        """Initialize."""
        self.tile = tile
        self.path = path
        self.nodata = nodata
        self._cache = {}

    def read(self, masked=True):
        """Read reprojected and resampled numpy array for current Tile."""
        if "data" not in self._cache:
            if not os.path.isfile(self.path):
                self._cache["data"] = None
            else:
                data = bp.unpack_ndarray_file(self.path)
                if masked:
                    self._cache["data"] = ma.masked_where(
                        data == self.nodata, data)
                else:
                    self._cache["data"] = data
        return self._cache["data"]

    def is_empty(self):
        """Check if there is data within this tile."""
        if not os.path.isfile(self.path):
            return True
        data = self.read()
        if isinstance(self.read(), ma.masked_array) and data.mask.all():
            return True
        elif isinstance(data, np.ndarray) and (
            np.where(data == self.nodata, True, False).all()
        ):
            return True
        else:
            return False

    def __enter__(self):
        """Enable context manager."""
        return self

    def __exit__(self, t, v, tb):
        """Clear cache on close."""
        del self._cache
