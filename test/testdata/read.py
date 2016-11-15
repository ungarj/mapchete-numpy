#!/usr/bin/env python

from mapchete import MapcheteProcess

class Process(MapcheteProcess):
    """Main process class"""
    def __init__(self, **kwargs):
        """Process initialization"""
        # init process
        MapcheteProcess.__init__(self, **kwargs)
        self.identifier = "my_process_id",
        self.title="My long process title",
        self.version = "0.1",
        self.abstract="short description on what my process does"

    def execute(self):
        with self.open("file1") as test_file:
            assert not test_file.is_empty()
            data = test_file.read()
            assert not data.mask.all()
        return data[0]
