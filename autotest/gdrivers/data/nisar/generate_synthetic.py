#!/usr/bin/env python
###############################################################################
#
# Project:  GDAL/OGR Test Suite
# Purpose:  Generate the synthetic NISAR granules used by gdrivers/nisar.py
# Author:   NISAR project, Jet Propulsion Laboratory
#
###############################################################################
# Copyright 2025 California Institute of Technology
# U.S. Government sponsorship acknowledged.
#
# SPDX-License-Identifier: MIT
###############################################################################

# Requires h5py and numpy. Run from this directory:
#
#   python generate_synthetic.py
#
# The constants below are duplicated in gdrivers/nisar.py: keep them in sync.

import h5py
import numpy as np

EPOCH = "seconds since 2025-01-01T00:00:00"


def _identification(f, product, level):
    ident = f.create_group("/science/LSAR/identification")
    ident.create_dataset("productType", data=np.bytes_(product))
    ident.create_dataset("productLevel", data=np.bytes_(level))


def gcov():
    """L2 GCOV: two Float32 terms, a mask and a radarGrid metadata cube."""

    nrows, ncols = 20, 30
    x0, dx, y0, dy = 500010.0, 20.0, 3799990.0, -20.0
    grid = "/science/LSAR/GCOV/grids/frequencyA"
    cube = "/science/LSAR/GCOV/metadata/radarGrid"

    with h5py.File("NISAR_L2_GCOV_synthetic.h5", "w", libver="earliest") as f:
        _identification(f, "GCOV", "L2")

        g = f.create_group(grid)
        g.create_dataset("listOfCovarianceTerms", data=np.array([b"HHHH", b"HVHV"]))
        for name, scale in (("HHHH", 1), ("HVHV", 2)):
            data = np.arange(nrows * ncols, dtype=np.float32) * scale
            g.create_dataset(
                name,
                data=data.reshape(nrows, ncols),
                chunks=(10, 15),
                compression="gzip",
            )
        # 0 = invalid, 1..5 = valid sub-swath number, 255 = fill
        mask = np.ones((nrows, ncols), dtype=np.uint8)
        mask[0:3, :] = 0
        mask[:, 0:2] = 255
        mask[10:, 15:] = 5
        g.create_dataset("mask", data=mask, chunks=(10, 15), compression="gzip")
        g.create_dataset("xCoordinates", data=x0 + dx * np.arange(ncols))
        g.create_dataset("yCoordinates", data=y0 + dy * np.arange(nrows))
        proj = g.create_dataset("projection", data=np.uint32(32611))
        proj.attrs["epsg_code"] = np.uint32(32611)

        # Coarse metadata cube (height, y, x) enclosing the imaging grid.
        # incidenceAngle is linear in x, y and height.
        c = f.create_group(cube)
        cx = 499900.0 + 200.0 * np.arange(5)
        cy = 3800100.0 - 200.0 * np.arange(4)
        cz = np.array([0.0, 1000.0, 2000.0])
        c.create_dataset("xCoordinates", data=cx)
        c.create_dataset("yCoordinates", data=cy)
        c.create_dataset("heightAboveEllipsoid", data=cz)
        proj = c.create_dataset("projection", data=np.uint32(32611))
        proj.attrs["epsg_code"] = np.uint32(32611)
        zz, yy, xx = np.meshgrid(cz, cy, cx, indexing="ij")
        inc = 30.0 + (xx - cx[0]) / 100.0 + (cy[0] - yy) / 400.0 + zz / 500.0
        # Chunked and compressed, as in real products
        c.create_dataset(
            "incidenceAngle",
            data=inc.astype(np.float32),
            chunks=(1, 4, 5),
            compression="gzip",
        )


def rslc():
    """L1 RSLC: two CFloat32 swaths and a geolocationGrid (GCPs + cube)."""

    nlines, npixels = 24, 32
    root = "/science/LSAR/RSLC"

    with h5py.File("NISAR_L1_RSLC_synthetic.h5", "w", libver="earliest") as f:
        _identification(f, "RSLC", "L1")

        s = f.create_group(root + "/swaths")
        t = s.create_dataset("zeroDopplerTime", data=1000.0 + 0.5 * np.arange(nlines))
        t.attrs["units"] = np.bytes_(EPOCH)
        s.create_dataset("zeroDopplerTimeSpacing", data=np.float64(0.5))

        g = s.create_group("frequencyA")
        g.create_dataset("listOfPolarizations", data=np.array([b"HH", b"VV"]))
        g.create_dataset("slantRange", data=800000.0 + 10.0 * np.arange(npixels))
        g.create_dataset("slantRangeSpacing", data=np.float64(10.0))
        line, pixel = np.mgrid[0:nlines, 0:npixels]
        hh = (pixel + 1j * line).astype(np.complex64)
        for name, data in (("HH", hh), ("VV", np.conj(hh) * 2)):
            g.create_dataset(name, data=data, chunks=(12, 16), compression="gzip")

        # geolocationGrid (height, azimuth time, slant range) enclosing the
        # swath, in EPSG:4326. incidenceAngle is linear along the three axes.
        c = f.create_group(root + "/metadata/geolocationGrid")
        c.create_dataset("epsg", data=np.int32(4326))
        cz = np.array([0.0, 1000.0, 2000.0])
        ct = 1000.0 + 5.75 * np.arange(3)
        cr = 800000.0 + 155.0 * np.arange(3)
        c.create_dataset("heightAboveEllipsoid", data=cz)
        t = c.create_dataset("zeroDopplerTime", data=ct)
        t.attrs["units"] = np.bytes_(EPOCH)
        c.create_dataset("slantRange", data=cr)
        k, i, j = np.meshgrid(np.arange(3), np.arange(3), np.arange(3), indexing="ij")
        inc = 30.0 + 5.0 * j + 1.0 * i + 0.5 * k
        for name, data in (
            ("coordinateX", -118.0 + 0.01 * j + 0.001 * k),
            ("coordinateY", 34.0 + 0.01 * i),
            ("incidenceAngle", inc.astype(np.float32)),
        ):
            # Chunked and compressed, as in real products
            c.create_dataset(name, data=data, chunks=(1, 3, 3), compression="gzip")


if __name__ == "__main__":
    gcov()
    rslc()
