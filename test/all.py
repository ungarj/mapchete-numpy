#!/usr/bin/env python
"""Test NumPy read/write driver."""

import os
import sys
import yaml
import shutil
import argparse
from multiprocessing import Pool
from functools import partial
from mapchete import Mapchete
from mapchete.config import MapcheteConfig
from mapchete.formats import available_input_formats, load_input_reader


def main(args):
    """Run all tests."""
    assert "NumPy" in available_input_formats()

    scriptdir = os.path.dirname(os.path.realpath(__file__))
    mapchete_file = os.path.join(scriptdir, "testdata/numpy.mapchete")

if __name__ == "__main__":
    main(sys.argv)
