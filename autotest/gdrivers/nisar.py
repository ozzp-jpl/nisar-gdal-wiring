#!/usr/bin/env pytest
# -*- coding: utf-8 -*-
###############################################################################
#
# Project:  GDAL/OGR Test Suite
# Purpose:  Test read functionality for NISAR driver.
# Author:   NISAR project, Jet Propulsion Laboratory
#
###############################################################################
# Copyright 2025 California Institute of Technology
# U.S. Government sponsorship acknowledged.
#
# SPDX-License-Identifier: MIT
###############################################################################

import array
import json
import math
import shutil
import struct

import gdaltest
import pytest
import webserver

from osgeo import gdal, osr

pytestmark = pytest.mark.require_driver("NISAR")

###############################################################################
# data/nisar/*.h5 are small synthetic granules written by
# data/nisar/generate_synthetic.py (h5py + numpy):
#
# - NISAR_L2_GCOV_synthetic.h5: L2 GCOV-like product with two Float32
#   covariance terms, a mask and a metadata/radarGrid incidenceAngle cube.
# - NISAR_L1_RSLC_synthetic.h5: L1 RSLC-like product with two CFloat32 swaths
#   and a metadata/geolocationGrid (GCPs + incidenceAngle cube).
#
# Tests that need real granules are marked require_run_on_demand + slow and
# fetch them from OSGeo/gdal-test-datasets (see REAL_GRANULES below).

GCOV = "data/nisar/NISAR_L2_GCOV_synthetic.h5"
NOT_NISAR = "data/hdf5/groups.h5"
GRID = "/science/LSAR/GCOV/grids/frequencyA"
GCOV_CUBE = "/science/LSAR/GCOV/metadata/radarGrid"
NROWS, NCOLS = 20, 30
X0, DX = 500010.0, 20.0
Y0, DY = 3799990.0, -20.0
EPSG = 32611
CHECKSUM = {"HHHH": 6742, "HVHV": 6809}

RSLC = "data/nisar/NISAR_L1_RSLC_synthetic.h5"
SWATH = "/science/LSAR/RSLC/swaths/frequencyA"
RSLC_CUBE = "/science/LSAR/RSLC/metadata/geolocationGrid"
NLINES, NPIXELS = 24, 32
RSLC_CHECKSUM = {"HH": 14384}

# Real granules, to be hosted in OSGeo/gdal-test-datasets (not yet uploaded:
# file names and expected values are to be filled in when they are).
REAL_GRANULES_URL = (
    "https://raw.githubusercontent.com/OSGeo/gdal-test-datasets/master/nisar/"
)
REAL_GRANULES = {
    "RSLC": "NISAR_L1_PR_RSLC_subset.h5",
    "GSLC": "NISAR_L2_PR_GSLC_subset.h5",
    "GCOV": "NISAR_L2_PR_GCOV_subset.h5",
    "GUNW": "NISAR_L2_PR_GUNW_subset.h5",
}


def _expected_data(scale):
    return array.array("f", [i * scale for i in range(NROWS * NCOLS)]).tobytes()


@pytest.fixture(autouse=True)
def check_no_file_leaks():
    num_files = len(gdaltest.get_opened_files())

    yield

    diff = len(gdaltest.get_opened_files()) - num_files
    assert diff <= 0, "Leak of file handles: %d leaked" % diff


###############################################################################
# Identify: NISAR granule with and without the NISAR: prefix, and a plain
# HDF5 file that must be left to the HDF5 driver.


def test_nisar_identify():

    drv = gdal.IdentifyDriverEx(GCOV, allowed_drivers=["NISAR"])
    assert drv is not None and drv.ShortName == "NISAR"

    drv = gdal.IdentifyDriver("NISAR:" + GCOV)
    assert drv is not None and drv.ShortName == "NISAR"

    assert gdal.IdentifyDriverEx(NOT_NISAR, allowed_drivers=["NISAR"]) is None


def test_nisar_open_non_nisar_fails():

    with pytest.raises(Exception):
        gdal.OpenEx(NOT_NISAR, allowed_drivers=["NISAR"])

    with pytest.raises(Exception):
        gdal.Open("NISAR:" + NOT_NISAR)

    with pytest.raises(Exception):
        gdal.Open(f'NISAR:"{GCOV}":{GRID}/does_not_exist')


def test_nisar_open_prefixed_without_extension(tmp_path):

    # An explicit NISAR: prefix claims the file whatever its extension, for
    # Open() as well as Identify().
    dst = tmp_path / "granule_without_extension"
    shutil.copy(GCOV, dst)

    assert gdal.IdentifyDriver(f'NISAR:"{dst}":{GRID}/HHHH').ShortName == "NISAR"
    ds = gdal.Open(f'NISAR:"{dst}":{GRID}/HHHH')
    assert ds.GetRasterBand(1).Checksum() == CHECKSUM["HHHH"]


###############################################################################
# Container open: no bands, subdataset inventory, metadata domains.


def test_nisar_subdatasets():

    ds = gdal.OpenEx(GCOV, allowed_drivers=["NISAR"])
    assert ds.GetDriver().ShortName == "NISAR"
    assert ds.RasterCount == 0

    sds = ds.GetSubDatasets()
    assert [name for name, _ in sds] == [
        f'NISAR:"{GCOV}":{GRID}/HHHH',
        f'NISAR:"{GCOV}":{GRID}/HVHV',
        f'NISAR:"{GCOV}":{GRID}/mask',
        f'NISAR:"{GCOV}":{GCOV_CUBE}/incidenceAngle',
    ]
    assert sds[3][1] == f"[3x4x5] {GCOV_CUBE}/incidenceAngle (Float32)"
    assert sds[0][1] == f"[{NROWS}x{NCOLS}] {GRID}/HHHH (Float32)"

    domains = ds.GetMetadataDomainList()
    assert "SUBDATASETS" in domains
    assert "NISAR_GLOBAL" in domains
    assert len(domains) == len(set(domains)), "duplicated metadata domains"

    # Repeated reads must return the same (dataset-owned) inventory
    assert ds.GetMetadata("SUBDATASETS") == ds.GetMetadata("SUBDATASETS")

    ds = None
    assert not gdaltest.is_file_open(GCOV), "file still opened."


###############################################################################
# Open a subdataset through the standard GDALTest harness (checksum,
# geotransform, projection).


@pytest.mark.parametrize("pol", ["HHHH", "HVHV"])
def test_nisar_gdaltest_open(pol):

    tst = gdaltest.GDALTest(
        "NISAR",
        f'NISAR:"{GCOV}":{GRID}/{pol}',
        1,
        CHECKSUM[pol],
        filename_absolute=1,
    )
    tst.testOpen(
        check_gt=(X0 - DX / 2, DX, 0, Y0 - DY / 2, 0, DY),
        check_prj=f"EPSG:{EPSG}",
    )


###############################################################################


@pytest.mark.parametrize(
    "connection",
    [
        f'NISAR:"{GCOV}":{GRID}/HVHV',
        f"NISAR:{GCOV}:{GRID}/HVHV",
    ],
    ids=["quoted", "unquoted"],
)
def test_nisar_open_subdataset(connection):

    ds = gdal.Open(connection)
    assert ds.GetDriver().ShortName == "NISAR"
    assert (ds.RasterXSize, ds.RasterYSize, ds.RasterCount) == (NCOLS, NROWS, 1)
    assert ds.GetMetadataItem("HDF5_PATH") == f"{GRID}/HVHV"

    band = ds.GetRasterBand(1)
    assert band.DataType == gdal.GDT_Float32
    assert band.ReadRaster() == _expected_data(2)
    assert band.Checksum() == CHECKSUM["HVHV"]


def test_nisar_read_window():

    ds = gdal.Open(f'NISAR:"{GCOV}":{GRID}/HHHH')
    band = ds.GetRasterBand(1)
    # Window straddling the 10x15 HDF5 chunk boundaries
    data = band.ReadRaster(12, 7, 6, 5)
    expected = array.array(
        "f", [float(y * NCOLS + x) for y in range(7, 12) for x in range(12, 18)]
    ).tobytes()
    assert data == expected


###############################################################################
# Georeferencing: xCoordinates / yCoordinates are pixel centers, SRS comes
# from the epsg_code attribute of the sibling "projection" dataset.


def test_nisar_georeferencing():

    ds = gdal.Open(f"NISAR:{GCOV}:{GRID}/HHHH")
    assert ds.GetGeoTransform() == pytest.approx(
        (X0 - DX / 2, DX, 0, Y0 - DY / 2, 0, DY)
    )
    srs = ds.GetSpatialRef()
    assert srs is not None
    assert srs.GetAuthorityName(None) == "EPSG"
    assert srs.GetAuthorityCode(None) == str(EPSG)
    assert ds.GetGCPCount() == 0


###############################################################################
# Open options


@pytest.mark.parametrize(
    "options,expected",
    [
        (["FREQ=A"], "HHHH"),
        (["INST=LSAR", "POL=HVHV"], "HVHV"),
        (["pol=hvhv"], "HVHV"),
    ],
)
def test_nisar_open_options(options, expected):

    ds = gdal.OpenEx(GCOV, allowed_drivers=["NISAR"], open_options=options)
    assert ds.GetMetadataItem("HDF5_PATH") == f"{GRID}/{expected}"
    assert ds.GetRasterBand(1).ReadRaster() == _expected_data(
        1 if expected == "HHHH" else 2
    )


@pytest.mark.parametrize(
    "options,message",
    [
        (["INST=XSAR"], "Invalid INST"),
        (["FREQ=C"], "Invalid FREQ"),
        (["POL=VVVV"], "Invalid POL"),
        (["DUMP_MODE=BOGUS"], "Invalid DUMP_MODE"),
        (["QUANTITY=incidenceAngle"], "DEM_FILE"),
    ],
)
def test_nisar_open_options_invalid(options, message):

    with pytest.raises(Exception, match=message):
        gdal.OpenEx(GCOV, allowed_drivers=["NISAR"], open_options=options)


###############################################################################
# Read-only driver


def test_nisar_update_rejected():

    with pytest.raises(Exception, match="only supports read access"):
        gdal.OpenEx(
            f"NISAR:{GCOV}:{GRID}/HHHH",
            gdal.OF_RASTER | gdal.OF_UPDATE,
            allowed_drivers=["NISAR"],
        )


def test_nisar_no_create():

    drv = gdal.GetDriverByName("NISAR")
    assert drv.GetMetadataItem(gdal.DCAP_CREATE) is None
    assert drv.GetMetadataItem(gdal.DCAP_CREATECOPY) is None
    assert drv.GetMetadataItem(gdal.DCAP_RASTER) == "YES"
    assert drv.GetMetadataItem(gdal.DMD_HELPTOPIC) == "drivers/raster/nisar.html"
    assert drv.GetMetadataItem(gdal.DMD_OPENOPTIONLIST) is not None


def test_nisar_driver_version():

    # Reported by in-tree builds as well as by the standalone plugin
    version = gdal.GetDriverByName("NISAR").GetMetadataItem("DRIVER_VERSION")
    assert version is not None
    assert version.startswith("v")
    assert len(version.split(" ")[0].split(".")) == 3


###############################################################################
# Virtual file systems: Identify() and Open() go through the VSI Virtual File
# Layer for any /vsi path, not only for /vsicurl/ and /vsis3/.


@pytest.fixture()
def vsimem_gcov(tmp_vsimem):

    filename = str(tmp_vsimem / "NISAR_L2_GCOV_synthetic.h5")
    with open(GCOV, "rb") as f:
        gdal.FileFromMemBuffer(filename, f.read())
    return filename


def test_nisar_vsimem(tmp_vsimem, vsimem_gcov):

    drv = gdal.IdentifyDriverEx(vsimem_gcov, allowed_drivers=["NISAR"])
    assert drv is not None and drv.ShortName == "NISAR"

    # Identify() looks at the content, not at the file name
    renamed = str(tmp_vsimem / "renamed.h5")
    with open(GCOV, "rb") as f:
        gdal.FileFromMemBuffer(renamed, f.read())
    drv = gdal.IdentifyDriverEx(renamed, allowed_drivers=["NISAR"])
    assert drv is not None and drv.ShortName == "NISAR"

    not_nisar = str(tmp_vsimem / "NISAR_lookalike.h5")
    with open(NOT_NISAR, "rb") as f:
        gdal.FileFromMemBuffer(not_nisar, f.read())
    assert gdal.IdentifyDriverEx(not_nisar, allowed_drivers=["NISAR"]) is None

    ds = gdal.OpenEx(vsimem_gcov, allowed_drivers=["NISAR"])
    assert ds.GetSubDatasets()[0][0] == f'NISAR:"{vsimem_gcov}":{GRID}/HHHH'

    ds = gdal.Open(f'NISAR:"{vsimem_gcov}":{GRID}/HVHV')
    assert ds.GetRasterBand(1).Checksum() == CHECKSUM["HVHV"]
    assert ds.GetGeoTransform() == pytest.approx(
        (X0 - DX / 2, DX, 0, Y0 - DY / 2, 0, DY)
    )


def test_nisar_vsizip(tmp_path):

    zip_filename = str(tmp_path / "granule.zip")
    f = gdal.VSIFOpenL(f"/vsizip/{zip_filename}/NISAR_L2_GCOV_synthetic.h5", "wb")
    with open(GCOV, "rb") as src:
        data = src.read()
    assert gdal.VSIFWriteL(data, 1, len(data), f) == len(data)
    gdal.VSIFCloseL(f)

    filename = f"/vsizip/{zip_filename}/NISAR_L2_GCOV_synthetic.h5"
    drv = gdal.IdentifyDriverEx(filename, allowed_drivers=["NISAR"])
    assert drv is not None and drv.ShortName == "NISAR"
    ds = gdal.Open(f'NISAR:"{filename}":{GRID}/HHHH')
    assert ds.GetRasterBand(1).Checksum() == CHECKSUM["HHHH"]


@pytest.fixture(scope="module")
def server():

    process, port = webserver.launch(handler=webserver.DispatcherHttpHandler)
    if port == 0:
        pytest.skip("cannot start HTTP server")

    import collections

    WebServer = collections.namedtuple("WebServer", "process port")

    yield WebServer(process, port)

    gdal.VSICurlClearCache()
    webserver.server_stop(process, port)


class _RangeFileHandler(webserver.FileHandler):
    """webserver.FileHandler with HTTP range requests on files of any size."""

    def send_response(self, request, filedata):
        if filedata is None:
            request.send_response(404)
            request.end_headers()
            return

        import re

        start, end = 0, len(filedata)
        res = re.search(r"bytes=(\d+)-(\d+)", request.headers.get("Range", ""))
        if res:
            start = int(res.group(1))
            end = min(int(res.group(2)) + 1, len(filedata))
            request.send_response(206)
            request.send_header(
                "Content-Range", "bytes %d-%d/%d" % (start, end - 1, len(filedata))
            )
        else:
            request.send_response(200)
        request.send_header("Content-Length", end - start)
        request.end_headers()
        request.wfile.write(filedata[start:end])


@pytest.mark.require_curl()
def test_nisar_vsicurl(server):

    gdal.VSICurlClearCache()

    files = {}
    for filename in (GCOV, RSLC):
        with open(filename, "rb") as f:
            files["/" + filename.split("/")[-1]] = f.read()
    handler = _RangeFileHandler(files)
    base = f"/vsicurl/http://localhost:{server.port}"

    with webserver.install_http_handler(handler):
        filename = f"{base}/NISAR_L2_GCOV_synthetic.h5"
        drv = gdal.IdentifyDriverEx(filename, allowed_drivers=["NISAR"])
        assert drv is not None and drv.ShortName == "NISAR"

        ds = gdal.OpenEx(filename, allowed_drivers=["NISAR"])
        assert ds.GetSubDatasets()[1][0] == f'NISAR:"{filename}":{GRID}/HVHV'
        ds = None

        ds = gdal.Open(f'NISAR:"{filename}":{GRID}/HVHV')
        assert ds.GetRasterBand(1).Checksum() == CHECKSUM["HVHV"]
        assert ds.GetSpatialRef().GetAuthorityCode(None) == str(EPSG)
        ds = None

        # http:// URLs are mapped to /vsicurl/
        url = f"http://localhost:{server.port}/NISAR_L1_RSLC_synthetic.h5"
        ds = gdal.Open(f'NISAR:"{url}":{SWATH}/HH')
        assert ds.GetRasterBand(1).Checksum() == RSLC_CHECKSUM["HH"]
        assert ds.GetGCPCount() == 9
        ds = None

    # Release the HTTP connection kept alive by /vsicurl/
    gdal.VSICurlClearCache()


###############################################################################
# L1 products: no geotransform, GCPs from metadata/geolocationGrid.


def test_nisar_l1_subdatasets():

    ds = gdal.OpenEx(RSLC, allowed_drivers=["NISAR"])
    sds = dict(ds.GetSubDatasets())
    assert (
        sds[f'NISAR:"{RSLC}":{SWATH}/HH']
        == f"[{NLINES}x{NPIXELS}] {SWATH}/HH (complex, Float32)"
    )
    assert f'NISAR:"{RSLC}":{SWATH}/VV' in sds
    assert f'NISAR:"{RSLC}":{RSLC_CUBE}/incidenceAngle' in sds


def test_nisar_l1_gcps():

    ds = gdal.Open(f'NISAR:"{RSLC}":{SWATH}/HH')
    assert (ds.RasterXSize, ds.RasterYSize) == (NPIXELS, NLINES)
    assert ds.GetGeoTransform(can_return_null=True) is None
    assert ds.GetSpatialRef() is None

    assert ds.GetGCPSpatialRef().GetAuthorityCode(None) == "4326"
    gcps = ds.GetGCPs()
    assert ds.GetGCPCount() == 9 and len(gcps) == 9

    # geolocationGrid is 3 (zeroDopplerTime) x 3 (slantRange) nodes. Node j is
    # at 155 m / 10 m = 15.5 pixels, node i at 5.75 s / 0.5 s = 11.5 lines;
    # pixel and line refer to pixel centres. Ground coordinates are those of
    # the lowest height layer.
    for i in range(3):
        for j in range(3):
            gcp = gcps[i * 3 + j]
            assert gcp.Id == str(i * 3 + j + 1)
            assert gcp.GCPPixel == pytest.approx(15.5 * j + 0.5)
            assert gcp.GCPLine == pytest.approx(11.5 * i + 0.5)
            assert gcp.GCPX == pytest.approx(-118.0 + 0.01 * j)
            assert gcp.GCPY == pytest.approx(34.0 + 0.01 * i)
            assert gcp.GCPZ == 0

    # A GCP transformer can be built from them
    tr = gdal.Transformer(ds, None, ["METHOD=GCP_POLYNOMIAL", "MAX_GCP_ORDER=1"])
    success, pnt = tr.TransformPoint(0, 16.0, 12.0)
    assert success
    assert pnt[0] == pytest.approx(-117.99, abs=1e-6)
    assert pnt[1] == pytest.approx(34.01, abs=1e-6)


def test_nisar_l1_open_options():

    ds = gdal.OpenEx(RSLC, allowed_drivers=["NISAR"], open_options=["POL=VV"])
    assert ds.GetMetadataItem("HDF5_PATH") == f"{SWATH}/VV"
    assert ds.GetGCPCount() == 9

    with pytest.raises(Exception, match="Invalid POL"):
        gdal.OpenEx(RSLC, allowed_drivers=["NISAR"], open_options=["POL=HV"])


###############################################################################
# Complex rasters and derived subdatasets


def _complex_at(band, pixel, line):
    return struct.unpack("ff", band.ReadRaster(pixel, line, 1, 1))


def test_nisar_complex():

    # HH = pixel + j * line, VV = 2 * conj(HH)
    ds = gdal.Open(f'NISAR:"{RSLC}":{SWATH}/HH')
    band = ds.GetRasterBand(1)
    assert band.DataType == gdal.GDT_CFloat32
    assert band.Checksum() == RSLC_CHECKSUM["HH"]
    assert _complex_at(band, 5, 3) == (5, 3)
    # Last pixel of the last 12x16 chunk
    assert _complex_at(band, NPIXELS - 1, NLINES - 1) == (NPIXELS - 1, NLINES - 1)

    ds = gdal.Open(f'NISAR:"{RSLC}":{SWATH}/VV')
    assert _complex_at(ds.GetRasterBand(1), 5, 3) == (10, -6)


@pytest.mark.require_driver("DERIVED")
def test_nisar_derived_subdatasets_complex():

    ds = gdal.Open(f'NISAR:"{RSLC}":{SWATH}/HH')
    assert "DERIVED_SUBDATASETS" in ds.GetMetadataDomainList()
    md = ds.GetMetadata("DERIVED_SUBDATASETS")
    names = {
        v.split(":")[1]: v
        for k, v in md.items()
        if k.startswith("DERIVED_SUBDATASET_") and k.endswith("_NAME")
    }
    assert set(names) >= {"AMPLITUDE", "PHASE", "REAL", "IMAG", "INTENSITY", "CONJ"}
    ds = None

    def value(kind):
        derived = gdal.Open(names[kind])
        assert (derived.RasterXSize, derived.RasterYSize) == (NPIXELS, NLINES)
        # Georeferencing is inherited from the NISAR raster
        assert derived.GetGCPCount() == 9
        band = derived.GetRasterBand(1)
        if kind == "CONJ":
            return struct.unpack(
                "ff", band.ReadRaster(5, 3, 1, 1, buf_type=gdal.GDT_CFloat32)
            )
        return struct.unpack(
            "f", band.ReadRaster(5, 3, 1, 1, buf_type=gdal.GDT_Float32)
        )[0]

    # HH(5, 3) = 5 + 3j
    assert value("AMPLITUDE") == pytest.approx(math.hypot(5, 3))
    assert value("PHASE") == pytest.approx(math.atan2(3, 5))
    assert value("REAL") == 5
    assert value("IMAG") == 3
    assert value("INTENSITY") == 34
    assert value("CONJ") == (5, -3)


@pytest.mark.require_driver("DERIVED")
def test_nisar_derived_subdatasets_real():

    ds = gdal.Open(f'NISAR:"{GCOV}":{GRID}/HHHH')
    md = ds.GetMetadata("DERIVED_SUBDATASETS")
    names = [v for k, v in md.items() if k.endswith("_NAME")]
    assert names == [f'DERIVED_SUBDATASET:LOGAMPLITUDE:NISAR:"{GCOV}":{GRID}/HHHH']
    ds = None

    derived = gdal.Open(names[0])
    assert derived.GetGeoTransform() == pytest.approx(
        (X0 - DX / 2, DX, 0, Y0 - DY / 2, 0, DY)
    )
    data = struct.unpack(
        "f", derived.GetRasterBand(1).ReadRaster(10, 0, 1, 1, buf_type=gdal.GDT_Float32)
    )
    assert data[0] == pytest.approx(1.0)  # log10(|10|)

    # The container has no raster, hence no derived subdatasets
    ds = gdal.OpenEx(GCOV, allowed_drivers=["NISAR"])
    assert "DERIVED_SUBDATASETS" not in ds.GetMetadataDomainList()


###############################################################################
# Mask band: sibling "mask" dataset, exposed on request (MASK=YES). Values
# 1..5 are valid; 0 and the 255 fill value are not.


def test_nisar_mask_default_off():

    ds = gdal.Open(f'NISAR:"{GCOV}":{GRID}/HHHH')
    band = ds.GetRasterBand(1)
    assert band.GetMaskFlags() == gdal.GMF_ALL_VALID
    assert band.GetMaskBand().ReadRaster() == b"\xff" * (NROWS * NCOLS)

    ds = gdal.OpenEx(f'NISAR:"{GCOV}":{GRID}/HHHH', open_options=["MASK=NO"])
    assert ds.GetRasterBand(1).GetMaskFlags() == gdal.GMF_ALL_VALID


@pytest.mark.parametrize("pol", ["HHHH", "HVHV"])
def test_nisar_mask(pol):

    ds = gdal.OpenEx(f'NISAR:"{GCOV}":{GRID}/{pol}', open_options=["MASK=YES"])
    band = ds.GetRasterBand(1)
    assert band.GetMaskFlags() == gdal.GMF_PER_DATASET

    mask = band.GetMaskBand()
    assert mask.DataType == gdal.GDT_Byte
    assert (mask.XSize, mask.YSize) == (NCOLS, NROWS)

    # Rows 0..2 are 0 (invalid), columns 0..1 are 255 (fill), the rest is
    # 1 or 5 (valid sub-swaths)
    expected = bytes(
        0 if (y < 3 or x < 2) else 255 for y in range(NROWS) for x in range(NCOLS)
    )
    assert mask.ReadRaster() == expected
    # Window straddling the 10x15 chunk boundaries
    assert mask.ReadRaster(1, 2, 16, 10) == bytes(
        0 if (y < 3 or x < 2) else 255 for y in range(2, 12) for x in range(1, 17)
    )

    # The data itself is unchanged
    assert band.Checksum() == CHECKSUM[pol]


def test_nisar_mask_translate(tmp_vsimem):

    # The mask travels with the data through a format conversion
    src = gdal.OpenEx(f'NISAR:"{GCOV}":{GRID}/HHHH', open_options=["MASK=YES"])
    out = gdal.Translate(tmp_vsimem / "out.tif", src)
    assert out.GetRasterBand(1).GetMaskFlags() == gdal.GMF_PER_DATASET
    assert (
        out.GetRasterBand(1).GetMaskBand().Checksum()
        == src.GetRasterBand(1).GetMaskBand().Checksum()
    )


def test_nisar_mask_missing():

    # No sibling "mask" dataset in the RSLC fixture: all valid
    ds = gdal.OpenEx(f'NISAR:"{RSLC}":{SWATH}/HH', open_options=["MASK=YES"])
    band = ds.GetRasterBand(1)
    assert band.GetMaskFlags() == gdal.GMF_ALL_VALID
    assert band.GetMaskBand().ReadRaster() == b"\xff" * (NLINES * NPIXELS)


###############################################################################
# Metadata cubes: (height, y, x) datasets are exposed with one band per
# height layer, and interpolated onto the imaging grid at the DEM height with
# QUANTITY + DEM_FILE.


def _create_dem(filename, epsg, geotransform, height, size=40):

    ds = gdal.GetDriverByName("GTiff").Create(filename, size, size, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetSpatialRef(srs)
    ds.GetRasterBand(1).Fill(height)
    ds.Close()
    return str(filename)


def test_nisar_metadata_cube():

    ds = gdal.Open(f'NISAR:"{GCOV}":{GCOV_CUBE}/incidenceAngle')
    assert (ds.RasterXSize, ds.RasterYSize, ds.RasterCount) == (5, 4, 3)
    # Cube posting is 200 m, first node (pixel centre) at 499900, 3800100
    assert ds.GetGeoTransform() == pytest.approx((499800, 200, 0, 3800200, 0, -200))
    assert ds.GetSpatialRef().GetAuthorityCode(None) == str(EPSG)

    for i, height in enumerate((0, 1000, 2000)):
        band = ds.GetRasterBand(i + 1)
        assert float(band.GetMetadataItem("Z_VALUE")) == height
        # incidenceAngle = 30 + (x - 499900) / 100 + (3800100 - y) / 400 + z / 500
        data = struct.unpack("f" * 20, band.ReadRaster())
        assert data[0] == pytest.approx(30 + height / 500)
        assert data[19] == pytest.approx(30 + 8 + 1.5 + height / 500)


@pytest.mark.parametrize("height", [0.0, 500.0, 2000.0])
def test_nisar_interpolation_l2(tmp_vsimem, height):

    dem = _create_dem(
        tmp_vsimem / "dem.tif", EPSG, (499000, 100, 0, 3801000, 0, -100), height
    )
    ds = gdal.OpenEx(
        GCOV,
        allowed_drivers=["NISAR"],
        open_options=["QUANTITY=incidenceAngle", f"DEM_FILE={dem}"],
    )
    assert ds.GetDriver().ShortName == "NISAR"

    # Output is on the imaging grid of the default raster
    assert (ds.RasterXSize, ds.RasterYSize, ds.RasterCount) == (NCOLS, NROWS, 1)
    assert ds.GetGeoTransform() == pytest.approx(
        (X0 - DX / 2, DX, 0, Y0 - DY / 2, 0, DY)
    )
    assert ds.GetSpatialRef().GetAuthorityCode(None) == str(EPSG)
    assert ds.GetMetadataItem("NISAR_QUANTITY") == "incidenceAngle"
    assert ds.GetMetadataItem("NISAR_CUBE_PATH") == f"{GCOV_CUBE}/incidenceAngle"
    assert ds.GetMetadataItem("NISAR_REFERENCE_GRID") == f"{GRID}/HHHH"
    assert ds.GetMetadataItem("NISAR_GRID_TYPE") == "GEOCODED"

    # The cube is linear along x, y and height: trilinear interpolation at
    # the pixel centres is exact.
    band = ds.GetRasterBand(1)
    assert band.DataType == gdal.GDT_Float32
    data = struct.unpack("f" * (NROWS * NCOLS), band.ReadRaster())
    for row, col in ((0, 0), (7, 12), (NROWS - 1, NCOLS - 1)):
        x = X0 + DX * col
        y = Y0 + DY * row
        expected = 30 + (x - 499900) / 100 + (3800100 - y) / 400 + height / 500
        assert data[row * NCOLS + col] == pytest.approx(expected, abs=1e-4)


def test_nisar_interpolation_l2_pol_and_reprojected_dem(tmp_vsimem):

    # DEM in another CRS than the cube: warped on the fly
    dem = _create_dem(
        tmp_vsimem / "dem.tif", 4326, (-117.5, 0.025, 0, 34.8, 0, -0.025), 1000.0
    )
    ds = gdal.OpenEx(
        GCOV,
        allowed_drivers=["NISAR"],
        open_options=["QUANTITY=incidenceAngle", "POL=HVHV", f"DEM_FILE={dem}"],
    )
    assert (ds.RasterXSize, ds.RasterYSize) == (NCOLS, NROWS)
    assert ds.GetMetadataItem("NISAR_REFERENCE_GRID") == f"{GRID}/HVHV"
    data = struct.unpack("f", ds.GetRasterBand(1).ReadRaster(12, 7, 1, 1))
    x = X0 + DX * 12
    y = Y0 + DY * 7
    assert data[0] == pytest.approx(
        30 + (x - 499900) / 100 + (3800100 - y) / 400 + 2, abs=1e-4
    )


def test_nisar_interpolation_l1(tmp_vsimem):

    dem = _create_dem(
        tmp_vsimem / "dem.tif", 4326, (-118.5, 0.025, 0, 34.5, 0, -0.025), 500.0
    )
    ds = gdal.OpenEx(
        RSLC,
        allowed_drivers=["NISAR"],
        open_options=["QUANTITY=incidenceAngle", f"DEM_FILE={dem}"],
    )

    # L1 output stays in radar coordinates, GCPs are passed through
    assert (ds.RasterXSize, ds.RasterYSize, ds.RasterCount) == (NPIXELS, NLINES, 1)
    assert ds.GetGeoTransform(can_return_null=True) is None
    assert ds.GetGCPCount() == 9
    assert ds.GetGCPSpatialRef().GetAuthorityCode(None) == "4326"
    assert ds.GetMetadataItem("NISAR_GRID_TYPE") == "RADAR"
    assert ds.GetMetadataItem("NISAR_CUBE_PATH") == f"{RSLC_CUBE}/incidenceAngle"
    assert ds.GetMetadataItem("NISAR_REFERENCE_GRID") == f"{SWATH}/HH"

    # incidenceAngle = 30 + 5 * range node + azimuth node + 0.5 * height node,
    # with nodes every 15.5 pixels, 11.5 lines and 1000 m.
    data = struct.unpack("f" * (NLINES * NPIXELS), ds.GetRasterBand(1).ReadRaster())
    for line, pixel in ((0, 0), (3, 5), (NLINES - 1, NPIXELS - 1)):
        expected = 30 + 5 * pixel / 15.5 + line / 11.5 + 0.5 * 0.5
        assert data[line * NPIXELS + pixel] == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(
    "filename,options,message",
    [
        (GCOV, ["QUANTITY=incidenceAngle"], "DEM_FILE"),
        (
            GCOV,
            ["QUANTITY=incidenceAngle", "DEM_FILE=/i_do/not_exist.tif"],
            None,
        ),
        (GCOV, ["QUANTITY=doesNotExist", "DEM_FILE=data/byte.tif"], None),
    ],
)
def test_nisar_interpolation_errors(filename, options, message):

    with pytest.raises(Exception, match=message):
        gdal.OpenEx(filename, allowed_drivers=["NISAR"], open_options=options)


###############################################################################
# Kerchunk sidecar: written on request, in a user-given directory.


def test_nisar_export_zarr_dir(tmp_path):

    with gdal.config_option("NISAR_EXPORT_ZARR_DIR", str(tmp_path)):
        ds = gdal.Open(f'NISAR:"{GCOV}":{GRID}/HHHH')
        assert ds.GetRasterBand(1).Checksum() == CHECKSUM["HHHH"]
        ds = None

    files = [p.name for p in tmp_path.iterdir()]
    assert files == ["nisar_kerchunk_science_LSAR_GCOV_grids_frequencyA_HHHH.json"]
    with open(tmp_path / files[0]) as f:
        j = json.load(f)
    assert j["version"] == 1
    zarray = json.loads(j["refs"][f"{GRID[1:]}/HHHH/.zarray"])
    assert zarray["shape"] == [NROWS, NCOLS]
    assert zarray["chunks"] == [10, 15]
    assert zarray["dtype"] == "<f4"
    # 2 x 2 chunks, each a [uri, offset, size] reference into the granule
    for key in ("0.0", "0.1", "1.0", "1.1"):
        uri, offset, size = j["refs"][f"{GRID[1:]}/HHHH/{key}"]
        assert uri.endswith("NISAR_L2_GCOV_synthetic.h5")
        assert offset > 0 and size > 0


def test_nisar_export_zarr_requires_dir(tmp_path):

    # No default output location: nothing is written without a directory
    with gdal.config_option("NISAR_EXPORT_ZARR", "YES"), gdal.quiet_errors():
        gdal.ErrorReset()
        ds = gdal.Open(f'NISAR:"{GCOV}":{GRID}/HHHH')
        assert ds.GetRasterBand(1).Checksum() == CHECKSUM["HHHH"]
        assert "NISAR_EXPORT_ZARR_DIR" in gdal.GetLastErrorMsg()


###############################################################################
# Multi-threaded reads: one dataset per thread, as the lock around libhdf5
# calls (when libhdf5 is not thread-safe) must make this safe.


def test_nisar_multithreaded_read():

    import threading

    errors = []

    def read(filename, path, checksum):
        try:
            for _ in range(5):
                ds = gdal.Open(f'NISAR:"{filename}":{path}')
                assert ds.GetRasterBand(1).Checksum() == checksum
                ds = None
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=read, args=args)
        for args in (
            (GCOV, f"{GRID}/HHHH", CHECKSUM["HHHH"]),
            (GCOV, f"{GRID}/HVHV", CHECKSUM["HVHV"]),
            (RSLC, f"{SWATH}/HH", RSLC_CHECKSUM["HH"]),
        )
        * 2
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


###############################################################################
# Real granules (too large for the GDAL tree): run on demand only.


def _real_granule(product):

    filename = REAL_GRANULES[product]
    gdaltest.download_or_skip(REAL_GRANULES_URL + filename, filename)
    return "tmp/cache/" + filename


@pytest.mark.require_run_on_demand
@pytest.mark.slow()
@pytest.mark.parametrize("product", ["RSLC", "GSLC", "GCOV", "GUNW"])
def test_nisar_real_granule_subdatasets(product):

    filename = _real_granule(product)
    ds = gdal.OpenEx(filename, allowed_drivers=["NISAR"])
    sds = ds.GetSubDatasets()
    assert sds
    for name, _ in sds:
        sub = gdal.Open(name)
        assert sub.RasterCount >= 1
        if product == "RSLC":
            if "/swaths/" in name:
                assert sub.GetGCPCount() > 0
        elif "/grids/" in name:
            assert sub.GetSpatialRef() is not None
            assert sub.GetGeoTransform(can_return_null=True) is not None


@pytest.mark.require_run_on_demand
@pytest.mark.slow()
def test_nisar_real_granule_l1_gcps():

    filename = _real_granule("RSLC")
    ds = gdal.OpenEx(filename, allowed_drivers=["NISAR"], open_options=["POL=HH"])
    assert ds.GetRasterBand(1).DataType == gdal.GDT_CFloat32
    assert ds.GetGCPCount() > 0
    assert ds.GetGCPSpatialRef() is not None
    assert gdal.Transformer(ds, None, ["METHOD=GCP_TPS"]) is not None


@pytest.mark.require_run_on_demand
@pytest.mark.slow()
def test_nisar_real_granule_complex_derived():

    filename = _real_granule("GSLC")
    ds = gdal.OpenEx(filename, allowed_drivers=["NISAR"], open_options=["POL=HH"])
    assert ds.GetRasterBand(1).DataType == gdal.GDT_CFloat32
    names = [
        v
        for k, v in ds.GetMetadata("DERIVED_SUBDATASETS").items()
        if k.endswith("_NAME") and ":AMPLITUDE:" in v
    ]
    assert len(names) == 1
    amp = gdal.Open(names[0])
    assert amp.GetGeoTransform() == pytest.approx(ds.GetGeoTransform())
    assert amp.GetRasterBand(1).ComputeRasterMinMax()[0] >= 0


@pytest.mark.require_run_on_demand
@pytest.mark.slow()
@pytest.mark.parametrize("product", ["GCOV", "GUNW"])
def test_nisar_real_granule_mask(product):

    filename = _real_granule(product)
    sds = gdal.OpenEx(filename, allowed_drivers=["NISAR"]).GetSubDatasets()
    name = [n for n, _ in sds if "/grids/" in n and not n.endswith("/mask")][0]
    ds = gdal.OpenEx(name, open_options=["MASK=YES"])
    band = ds.GetRasterBand(1)
    assert band.GetMaskFlags() == gdal.GMF_PER_DATASET
    # Per the GCOV / GUNW mask encodings, both valid and invalid pixels exist
    assert band.GetMaskBand().ComputeRasterMinMax() == (0, 255)


@pytest.mark.require_run_on_demand
@pytest.mark.slow()
@pytest.mark.parametrize("product", ["RSLC", "GCOV"])
def test_nisar_real_granule_interpolation(tmp_vsimem, product):

    filename = _real_granule(product)
    # Constant-height global DEM: the result is the cube layer at that height
    dem = _create_dem(
        tmp_vsimem / "dem.tif", 4326, (-180, 1, 0, 90, 0, -1), 0.0, size=360
    )
    ds = gdal.OpenEx(
        filename,
        allowed_drivers=["NISAR"],
        open_options=["QUANTITY=incidenceAngle", f"DEM_FILE={dem}"],
    )
    assert ds.GetMetadataItem("NISAR_GRID_TYPE") == (
        "RADAR" if product == "RSLC" else "GEOCODED"
    )
    vmin, vmax = ds.GetRasterBand(1).ComputeRasterMinMax()
    assert 0 < vmin < vmax < 90


@pytest.mark.require_run_on_demand
@pytest.mark.slow()
@pytest.mark.require_curl()
def test_nisar_real_granule_vsicurl():

    filename = REAL_GRANULES["GCOV"]
    url = "/vsicurl/" + REAL_GRANULES_URL + filename
    if gdal.VSIStatL(url) is None:
        pytest.skip(f"{url} not available")
    ds = gdal.OpenEx(url, allowed_drivers=["NISAR"])
    sds = ds.GetSubDatasets()
    assert sds
    sub = gdal.Open(sds[0][0])
    assert sub.GetSpatialRef() is not None
    assert sub.GetRasterBand(1).ReadRaster(0, 0, 16, 16) is not None
