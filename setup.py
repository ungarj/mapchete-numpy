#!/usr/bin/env python

from setuptools import setup

setup(
    name='mapchete-numpy',
    version='0.1',
    description='Mapchete NumPy read/write extension',
    author='Joachim Ungar',
    author_email='joachim.ungar@gmail.com',
    url='https://github.com/ungarj/mapchete-safe',
    license='MIT',
    packages=['mapchete_numpy'],
    install_requires=[
        'mapchete>=0.4',
        'bloscpack>=0.11.0',
        'blosc>=1.4.4'
        ],
    entry_points={'mapchete.formats.drivers': ['numpy=mapchete_numpy']},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ]
)
