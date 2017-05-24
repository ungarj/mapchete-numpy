"""
Handles writing process output into a pyramid of NumPy array files.

output configuration parameters
-------------------------------

mandatory
~~~~~~~~~

bands: integer
    number of output bands to be written
path: string
    output directory
dtype: string
    numpy datatype
ndim: integer
    number of dimensions
"""

import os
import numpy as np
import numpy.ma as ma

from mapchete.formats import base
from mapchete.tile import BufferedTile
from mapchete.io.raster import prepare_array, extract_from_tile


class OutputData(base.OutputData):
    """
    Template class handling process output data.

    Parameters
    ----------
    output_params : dictionary
        output parameters from Mapchete file

    Attributes
    ----------
    path : string
        path to output directory
    file_extension : string
        file extension for output files (.tif)
    output_params : dictionary
        output parameters from Mapchete file
    nodata : integer or float
        nodata value used when writing GeoTIFFs
    pixelbuffer : integer
        buffer around output tiles
    pyramid : ``tilematrix.TilePyramid``
        output ``TilePyramid``
    crs : ``rasterio.crs.CRS``
        object describing the process coordinate reference system
    srid : string
        spatial reference ID of CRS (e.g. "{'init': 'epsg:4326'}")
    """

    METADATA = {
        "driver_name": "NumPy",
        "data_type": "raster",
        "mode": "rw"
    }

    def __init__(self, output_params):
        """Initialize."""
        super(OutputData, self).__init__(output_params)
        self.path = output_params["path"]
        self.file_extension = ".np"
        self.output_params = output_params
        try:
            self.nodata = output_params["nodata"]
        except KeyError:
            self.nodata = 0
        try:
            self.single_file = output_params["single_file"]
        except KeyError:
            self.single_file = True
        if self.single_file == False:
            raise NotImplementedError()

    def read(self, output_tile):
        """
        Read existing process output.

        Parameters
        ----------
        output_tile : ``BufferedTile``
            must be member of output ``TilePyramid``

        Returns
        -------
        process output : ``BufferedTile`` with appended data
        """
        try:
            output_tile.data = np.load(self.get_path(output_tile))
        except IOError:
            output_tile.data = self.emtpy(output_tile)
        return output_tile

    def write(self, process_tile):
        """
        Write data from process tiles into NumPy array dumps.

        Parameters
        ----------
        process_tile : ``BufferedTile``
            must be member of process ``TilePyramid``
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        process_tile.data = prepare_array(
            process_tile.data, masked=True, nodata=self.nodata,
            dtype=self.output_params["dtype"]
        )
        # Convert from process_tile to output_tiles
        for tile in self.pyramid.intersecting(process_tile):
            out_tile = BufferedTile(tile, self.pixelbuffer)
            out_data = extract_from_tile(process_tile, out_tile)
            self.prepare_path(tile)
            out_data.dump(self.get_path(out_tile))

    def tiles_exist(self, process_tile):
        """
        Check whether all output tiles of a process tile exist.

        Parameters
        ----------
        process_tile : ``BufferedTile``
            must be member of process ``TilePyramid``

        Returns
        -------
        exists : bool
        """
        return all(
            os.path.exists(self.get_path(tile))
            for tile in self.pyramid.intersecting(process_tile)
        )

    def is_valid_with_config(self, config):
        """
        Check if output format is valid with other process parameters.

        Parameters
        ----------
        config : dictionary
            output configuration parameters

        Returns
        -------
        is_valid : bool
        """
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
        """
        Determine target file path.

        Parameters
        ----------
        tile : ``BufferedTile``
            must be member of output ``TilePyramid``

        Returns
        -------
        path : string
        """
        return os.path.join(*[
            self.path, str(tile.zoom), str(tile.row),
            str(tile.col) + self.file_extension]
        )

    def prepare_path(self, tile):
        """
        Create directory and subdirectory if necessary.

        Parameters
        ----------
        tile : ``BufferedTile``
            must be member of output ``TilePyramid``
        """
        try:
            os.makedirs(os.path.dirname(self.get_path(tile)))
        except OSError:
            pass

    def empty(self, process_tile):
        """
        Return empty data.

        Parameters
        ----------
        process_tile : ``BufferedTile``
            must be member of process ``TilePyramid``

        Returns
        -------
        empty data : array
            empty array with data type provided in output profile
        """
        return ma.masked_array(
            data=np.array(
                np.full(
                    (self.output_params["bands"], ) + process_tile.shape,
                    self.nodata,
                    dtype=self.output_params["dtype"]),
                ndmin=self.output_params["ndim"]
            ),
            mask=True
        )

    def open(self, tile, process, **kwargs):
        """
        Open process output as input for other process.

        Parameters
        ----------
        tile : ``Tile``
        process : ``MapcheteProcess``
        kwargs : keyword arguments
        """
        try:
            resampling = kwargs["resampling"]
        except KeyError:
            resampling = None
        return InputTile(tile, process, resampling)


class InputTile(base.InputTile):
    """
    Target Tile representation of input data.

    Parameters
    ----------
    tile : ``Tile``
    process : ``MapcheteProcess``
    resampling : string
        rasterio resampling method

    Attributes
    ----------
    tile : ``Tile``
    process : ``MapcheteProcess``
    resampling : string
        rasterio resampling method
    pixelbuffer : integer
    """

    def __init__(self, tile, process, resampling):
        """Initialize."""
        self.tile = tile
        self.process = process
        self.pixelbuffer = None
        self.resampling = resampling
        self._np_cache = None

    def read(self):
        """
        Read reprojected & resampled input data.

        Parameters
        ----------
        indexes : integer or list
            band number or list of band numbers

        Returns
        -------
        data : array
        """
        if self._np_cache is None:
            tile = self.process.get_raw_output(self.tile)
            self._np_cache = tile.data
        return self._np_cache

    def is_empty(self):
        """
        Check if there is data within this tile.

        Returns
        -------
        is empty : bool
        """
        # empty if tile does not intersect with file bounding box
        src_bbox = self.process.config.process_area()
        tile_geom = self.tile.bbox
        if not tile_geom.intersects(src_bbox):
            return True

        # empty if source is empty
        return self.read().mask.all()

    def __enter__(self):
        """Enable context manager."""
        return self

    def __exit__(self, t, v, tb):
        """Clear cache on close."""
        del self._np_cache
