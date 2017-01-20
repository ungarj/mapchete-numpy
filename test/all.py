#!/usr/bin/env python
"""Test NumPy read/write driver."""

import os
import sys
import shutil
from mapchete import Mapchete
from mapchete.config import MapcheteConfig
from mapchete.formats import available_output_formats


def main(args):
    """Run all tests."""
    assert "NumPy" in available_output_formats()

    scriptdir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(scriptdir, "testdata/tmp")
    # mapchete_file = os.path.join(
    #     scriptdir, "testdata/write_singlefile.mapchete")
    # process = Mapchete(MapcheteConfig(mapchete_file))
    # for zoom in range(5):
    #     for tile in process.get_process_tiles(zoom):
    #         output = process.execute(tile)
    #         process.write(output)
    # mapchete_file = os.path.join(scriptdir, "testdata/read_singlefile.mapchete")
    # process = Mapchete(MapcheteConfig(mapchete_file))
    # for zoom in range(5):
    #     for tile in process.get_process_tiles(zoom):
    #         output = process.execute(tile)
    #         process.write(output)

    try:
        mapchete_file = os.path.join(scriptdir, "testdata/write.mapchete")
        process = Mapchete(MapcheteConfig(mapchete_file))
        for zoom in range(5):
            for tile in process.get_process_tiles(zoom):
                output = process.execute(tile)
                process.write(output)
        mapchete_file = os.path.join(scriptdir, "testdata/read.mapchete")
        process = Mapchete(MapcheteConfig(mapchete_file))
        for zoom in range(5):
            for tile in process.get_process_tiles(zoom):
                output = process.execute(tile)
                process.write(output)
    except:
        raise
    finally:
        try:
            pass
            # shutil.rmtree(out_dir)
        except:
            pass


if __name__ == "__main__":
    main(sys.argv)
