# Authors: Sara Biddle
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import datetime as dt
import re as re

import numpy as np

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import create_info
from ..._fiff.utils import _mult_cal_one
from ...annotations import Annotations
from ...utils import (
    _check_fname,
    fill_doc,
    logger,
    verbose,
    warn
)
from ..base import BaseRaw
from ..nirx.nirx import _read_csv_rows_cols

@fill_doc
def read_raw_portamon(
    fname, preload=False, verbose=None
) -> "RawPortaMon":
    """Reader for an optical imaging recording.

    Parameters
    ----------
    fname : path-like
        Path to the PortaMon data file.
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawPortaMon
        A Raw object containing PortaMon data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawPortaMon.

    Notes
    -----
    Currently only reads files exported from Oxysoft as text files.
    Currently only supports PortaMon device configuration.
    """
    return RawPortaMon(fname, preload, verbose)

@fill_doc
class RawPortaMon(BaseRaw):
    """Raw object from PortaMon optical imaging file.

    Parameters
    ----------
    fname : path-like
        Path to the PortaMon data file
        %(preload)s
        %(verbose)s
    
    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    """

    @verbose
    def __init__(self, fname, preload=False, verbose=None):
        logger.info("Loading %s" % fname)

        strfname = str(fname)
        if strfname.endswith(".xml"):
            _read_xml_portamon(fname)
        elif strfname.endswith(".txt"):
            info, raw_extras, annot = _read_txt_portamon(fname)
        else:
            pass

        super().__init__(
            info,
            preload,
            filenames = [fname],
            first_samps = [raw_extras["first_samp_num"]],
            last_samps = [raw_extras["last_samp_num"]],
            raw_extras = [raw_extras],
            verbose = verbose,
        )

        self.set_annotations(annot)

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a segment of data from a file.

        PortaMon files exported from oxysoft have one row for each sample
        
        Columns in raw data file depend on 
        - self._raw_extras[fi]["optode-template"]
        - self._raw_extras[fi]["export_type"]
        """

        # get the line number that data starts in the file
        start_line = self._raw_extras[fi]["start_line"]
        # get the line number that data ends in the file
        end_line = self._raw_extras[fi]["end_line"]
        # get the first sample number from the file
        first_sample_number = self._raw_extras[fi]["first_samp_num"]

        # determine the line number to start reading from
        this_start_read = start_line + (start - first_sample_number) 
        # determine the number of rows to read
        this_nrows = stop - start

        # make sure all data to be read exists in the file
        assert this_start_read >= start_line
        assert this_nrows <= (end_line - start_line + 1)

        this_data = np.genfromtxt(
            fname = self._filenames[fi],
            delimiter = "\t",
            skip_header = this_start_read,
            max_rows = this_nrows,
            usecols = self._raw_extras[fi]["mrk_col"]
        ).T

        _mult_cal_one(data, this_data, idx, cals, mult)
        return data


def _read_xml_portamon(fname):
    pass


def _read_txt_portamon(fname):
    # check fname is a real location
    fname = str(_check_fname(fname, "read", True, "fname"))

    # raw_extras holds all variable necessary for reading the data out of the file
    raw_extras = dict()
    raw_extras["fname"] = fname

    # create placeholder variables
    start_line = legend_start_line = sources_start_line = end_line = np.inf
    mrk_col = optode_template  = col_names =  ch_types = None

    
    fnirs_wavelengths = list() # keep track of wavelengths of sources
    first_samps = last_samps = np.inf
    onsets = list()
    annotdesc = list()
    meas_date = None
    sfreq = export_type = device_num = sources = detectors = None

    complete = False
    # Read file header and grab some info.
    with open(fname) as fid:
        line_num = 0
        i_line = fid.readline()
        while i_line:
            # strip trailing \n from string
            # do not strip before while statment is evaluated
            # some lines are blank and stripping the \n before evaluating the while loop
            # will exit the loop early
            i_line = i_line.rstrip("\n")
            # most lines will be data, so check that first
            if line_num >= start_line:
                assert len(col_names) > 0
                assert mrk_col is not None
                if end_line == line_num:
                    break
                crnt_line = i_line.rsplit("\t")
                if len(crnt_line[-1]) >= 1:
                    # if there's an event, 
                    # calculate the time in seconds since the first sample number
                    # by subtracting the first sample number by this sample number
                    # and dividing by sfreq 
                    # and appending to onsets
                    # Annotations must record onsets as time in seconds since start of data
                    # and appending the text description to annotdesc
                    onset_seconds = (int(crnt_line[0]) - first_samps) / sfreq
                    onsets.append(onset_seconds)
                    desc = crnt_line[-1].replace("\t", " ")
                    annotdesc.append(desc)                        
            # now proceed with standard header parsing
            # Extract measurement date and time
            elif "Start of measurement" in i_line:
                try:
                    datetime_strlong = i_line.rsplit("\t")[1]
                    datetime_str = datetime_strlong.rsplit(".")[0]
                    # All PortaMon files are exported from Oxysoft with the format 
                    # YYYY-mm-dd HH:MM:SS.fff
                    dt_code = "%Y-%m-%d %H:%M:%S"
                    try:
                        meas_date = dt.datetime.strptime(datetime_str, dt_code)
                    except ValueError:
                        pass
                    else: 
                        meas_date = meas_date.replace(tzinfo=dt.timezone.utc)
                except Exception:
                    warn(
                        "Extraction of measurement date from PortaMon file failed. "
                        "This can be caused by files saved in certain locales "
                        f"(currently only dates formatted as {dt_code} supported). "
                        "Please report this as a github issue. "
                        "The date is being set to January 1st, 2000, "
                        f"instead of {repr(datetime_str)}"
                    )
            elif "Optode-template" in i_line:
                optode_template = i_line.rsplit("\t")[1]
                # Check the optode-template is supported
                if optode_template not in ["PortaMon TSI", "PortaMon TSI Fit Factor"]:
                    raise RuntimeError(
                        "MNE has not be tested with Oxysoft "
                        "Optode-template " % optode_template
                    )
            elif "Optode distance" in i_line:
                raw_extras["optode_distance"] = i_line.rsplit("\t")[1]
            elif "Optode gradients" in i_line:
                raw_extras["optode_gradients"] = i_line.rsplit("\t")[1]
            elif "DPF" in i_line:
                # DPF value used for data acquisition
                raw_extras["dpf"] = i_line.rsplit("\t")[1]
            elif "Device id" in i_line:
                # Device ID number to be saved as serial number in info["device_info"]
                device_num = i_line.rsplit("\t")[1]
            elif "time span (sample numbers)" in i_line:
                raw_extras["first_samp_num"]= first_samps = int(i_line.rsplit("\t")[1])
                raw_extras["last_samp_num"]= last_samps = int(i_line.rsplit("\t")[2]) - 1
            elif "Legend" in i_line:
                # Legend should start a couple lines later.
                legend_start_line = line_num + 1
                col_names = list()
            elif "Receivers" in i_line:
                detectors = int(i_line.rsplit("\t")[1])
            elif "Light sources" in i_line:
                sources = int(i_line.rsplit("\t")[1])
            elif "Light source wavelengths" in i_line:
                sources_start_line = line_num + 2
            elif "Export sample rate" in i_line:
                raw_extras["export_sample_rate"] = sfreq = float(i_line.rsplit("\t")[1])
                raw_extras["export_sample_rate_unit"] = i_line.rsplit("\t")[2]
            elif line_num >= sources_start_line and line_num < sources_start_line+6:
                fnirs_wavelengths.append(i_line.rsplit("\t")[2])
            elif line_num >= legend_start_line:
                if complete == False:
                    if "Event" in i_line:
                        # Data should start a couple lines later.
                        start_line = line_num + 4
                        # Data should have same number of lines as number_of_samples
                        end_line = start_line + (last_samps - first_samps)
                        col_names.append(i_line.rsplit("\t")[1])
                        complete = True
                    elif "ADC" in i_line:
                    # Data was exported from Oxysoft as Optical Densities
                        export_type = "optical densities"
                        mrk_col = [1,2,3,4,5,6]
                    elif "Trace" in i_line:
                    # Data was exported from Oxysoft after applying MBLL
                        export_type = "haemoglobin"
                    elif export_type == "haemoglobin":
                    # Data exported as haemoglobin has column names
                        col_names.append(i_line.rsplit("\t")[1])
                        mrk_col = [1,2,3,4,5,6]    
                    elif export_type == "optical densities":
                        if "Sample number" in i_line:
                            col_names.append(i_line.rsplit("\t")[1])
                        elif len(i_line) < 1:
                            pass
                        else: 
                            col_names.append(i_line.rsplit("\t")[3])
            line_num += 1
            i_line = fid.readline()
    assert sfreq is not None

    if meas_date is None:
        meas_date = dt.datetime(2000, 1,1,0,0,0, tzinfo = dt.timezone.utc)
    
    # Determine channel indices requested according to optode_template
    # and export type
    # if exported as optical densities, all columns have real data no matter the optode_template
    # if exported as haemoglobin, column data depends on optode template
    # TSIFit Factor optode template uses three sources
    # TSI optode template uses two sources

    snames = [f"S{source + 1}" for source in range(int(sources/2))]
    dnames = [f"D{det + 1}" for det in range(detectors)]
    sdnames = [s + "_" + d for d in dnames for s in snames]
    

    if export_type=="optical densities":
        ch_types = "fnirs_od"
        count = 0
        ch_names = list()
        for sd in sdnames:
            ch_names.append(sd + " " + fnirs_wavelengths[count])
            count += 1
            ch_names.append(sd + " " + fnirs_wavelengths[count])
            count += 1

    if export_type == "haemoglobin":
        ch_types = [
            "hbo",
            "hbr",
            "hbo",
            "hbr",
            "hbo",
            "hbr"
            ]
        ch_names = [sd + " " + hb for sd in sdnames for hb in ("hbo", "hbr")]

    raw_extras.update(start_line = start_line,
                       end_line = end_line,
                       optode_template = optode_template,
                       export_type = export_type,
                       col_names = col_names,
                       mrk_col = mrk_col)

    # Create info structure.
    info = create_info(ch_names, sfreq, ch_types)
    info.set_meas_date(meas_date)

    # add device to info
    info["device_info"] = {"model" : "PortaMon",
                           "serial" : device_num}

    #if optode_template == "TSI":
    #        info["bads"].extend([])
    

    # Store channel, source, and detector locations
    # The channel location is stored in the first 3 entries of loc.
    # The source location is stored in the second 3 entries of loc.
    # The detector location is stored in the third 3 entries of loc.
    # Also encode the light frequency in the structure.
    #for ch in ch_names:
    #   pass


    # create annotations from events
    # Events are taken from the events colums
    # where onsets is the sample number of the row
    # and annotdesc is the text in the (Event) column
    # duration is set to be 1/sfreq
    # because the (Event) column contains no information about duration, just onset
    # ch_names is set to None in order to apply to all channels
    annot = Annotations(onsets, 1/sfreq, annotdesc, ch_names = None)

    return info, raw_extras, annot