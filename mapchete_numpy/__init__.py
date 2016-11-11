"""Handles output pyramids using GeoTIFFS."""

import os
import numpy as np
import numpy.ma as ma

from mapchete.formats import base
from mapchete.tile import BufferedTile


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

    def write(self, process_tile, overwrite=False):
        """Write process output into GeoTIFFs."""
        raise NotImplementedError()

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

    def profile(self, tile):
        """Create a metadata dictionary for rasterio."""
        dst_metadata = GTIFF_PROFILE
        dst_metadata.pop("transform", None)
        dst_metadata.update(
            crs=tile.crs, width=tile.width, height=tile.height,
            affine=tile.affine, driver="GTiff",
            count=self.output_params["bands"],
            dtype=self.output_params["dtype"]
        )
        return dst_metadata

    def verify_data(self, tile):
        """Verify array data and move array into tuple if necessary."""
        try:
            assert isinstance(
                tile.data, (np.ndarray, ma.MaskedArray, tuple, list))
        except AssertionError:
            raise ValueError(
                "process output must be 2D NumPy array, masked array or a tuple"
                )
        try:
            if isinstance(tile.data, (tuple, list)):
                for band in tile.data:
                    assert band.ndim == 2
            else:
                assert tile.data.ndim in [2, 3]
        except AssertionError:
            raise ValueError(
                "each output band must be a 2D NumPy array")


    def prepare_data(self, data, profile):
        """
        Convert data into correct output.

        Returns a 3D masked NumPy array including all bands with the data type
        specified in the configuration.
        """
        if isinstance(data, (list, tuple)):
            out_data = ()
            out_mask = ()
            for band in data:
                if isinstance(band, ma.MaskedArray):
                    try:
                        assert band.shape == band.mask.shape
                        out_data += (band, )
                        out_mask += (band.mask, )
                    except:
                        out_data += (band.data, )
                        out_mask += (
                            np.where(band.data == self.nodata, True, False), )
                elif isinstance(band, np.ndarray):
                    out_data += (band)
                    out_mask += (np.where(band == self.nodata, True, False))
                else:
                    raise ValueError("input data bands must be NumPy arrays")
            assert len(out_data) == len(out_mask)
            return ma.MaskedArray(
                data=np.stack(out_data).astype(profile["dtype"]),
                mask=np.stack(out_mask))
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            data = ma.expand_dims(data, axis=0)
        if isinstance(data, ma.MaskedArray):
            try:
                assert data.shape == data.mask.shape
                return data.astype(profile["dtype"])
            except:
                return ma.MaskedArray(
                    data=data.astype(profile["dtype"]),
                    mask=np.where(band.data == self.nodata, True, False))
        elif isinstance(data, np.ndarray):
            return ma.MaskedArray(
                data=data.astype(profile["dtype"]),
                mask=np.where(data == self.nodata, True, False))


    def empty(self, process_tile):
        """Empty data."""
        profile = self.profile(process_tile)
        return ma.masked_array(
            data=np.full(
                (profile["count"], ) + process_tile.shape, profile["nodata"],
                dtype=profile["dtype"]),
            mask=True
        )
