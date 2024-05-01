# Authors: Sara Biddle
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime as dt

import numpy as np
import pytest
import scipy.io as spio
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less

from mne import pick_types
from mne.datasets import testing
from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_portamon
from mne.io.tests.test_raw import _test_raw_reader

testing_path = "C:/Users/sgb21f/Documents/NIRS_Study/NIRS Data/Testing"
# temporarily using local files for testing until testing files are available
portamon_tsiff_od = "C:/Users/sgb21f/Documents/NIRS_Study/NIRS Data/Testing/PortaMon_TSIFF_od_annot.txt"
portamon_tsi_hb = "C:/Users/sgb21f/Documents/NIRS_Study/NIRS Data/Testing/PortaMon_TSI_haemoglobin_annot.txt"

raw = read_raw_portamon(portamon_tsi_hb, preload = True)

def portamon_matches_snirf(portamon_snirf):
    """Test PortaMon raw files return same data as snirf."""
    raw, raw_snirf = portamon_snirf
    assert raw.ch_names == raw_snirf.ch_names

    assert_allclose(raw._data, raw_snirf._data)

# @requires_testing_data
def test_portamon():
    """Test PortaMon file."""

    raw = read_raw_portamon(portamon_tsiff_od, preload = True)

    # Test data import
    assert raw._data.shape == (6, 411)

    data = raw.get_data()
    assert data.shape == (6,411)
    
    # By default real data is returned
    assert np.sum(np.isnan(raw.get_data())) == 0


    raw = read_raw_portamon(portamon_tsi_hb, preload = True)

    assert raw._data.shape == (6, 711)

    data = raw.get_data()
    assert data.shape == (6,711)
    assert np.sum(np.isnan(raw.get_data())) == 0

    
# @requires_testing_data
def test_portamon_info():
    """Test header reading into RawPortaMon"""

    raw = read_raw_portamon(portamon_tsiff_od, preload = True)
    assert raw.info["sfreq"] == 10.00
    assert len(raw.annotations) == 3
    assert raw.annotations.description[0] == "annotation"
    # assert raw.annotations.onset[0] == 15.5

    assert raw.info["meas_date"] == dt.datetime(
        2024, 1, 3,  11, 23, 19, tzinfo = dt.timezone.utc
    )

    # Test channel naming
    assert raw.info["ch_names"][:6] == [
        "S1_D1 759",
        "S1_D1 854",
        "S2_D1 759",
        "S2_D1 854",
        "S3_D1 761",
        "S3_D1 856"
    ]

    raw = read_raw_portamon(portamon_tsi_hb, preload = True)
    assert raw.info["sfreq"] == 10.00
    assert len(raw.annotations) == 1
    assert raw.annotations.description[0] == "annotation"

    assert raw.info["meas_date"] == dt.datetime(
        2024,1,3,11,23,19, tzinfo = dt.timezone.utc
    )

    assert raw.info["ch_names"][:6] == [
        "S1_D1 hbo",
        "S1_D1 hbr",
        "S2_D1 hbo",
        "S2_D1 hbr",
        "S3_D1 hbo",
        "S3_D1 hbr"
    ]


    