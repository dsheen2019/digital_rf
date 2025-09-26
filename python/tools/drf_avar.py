#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Massachusetts Institute of Technology (MIT)
# All rights reserved.
#
# Distributed under the terms of the BSD 3-clause license.
#
# The full license is in the LICENSE file, distributed with this software.
# ----------------------------------------------------------------------------
"""compute the allan variance of the RF noise level for a drf recording to characterize telescope amplitude stability"""


import datetime
import argparse
import os
import sys
import psutil
import time
import traceback
import multiprocessing
from functools import partial

import dateutil
import digital_rf as drf
import matplotlib.gridspec
import matplotlib.mlab
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal



class AllenVarProcessor(object):
    def __init__(self, opt):
        """Initialize handler for the drf data"""
        self.opt = opt

        #general proceesing params

        available_memory_bytes = psutil.virtual_memory().available #definitely don't want to exceed this or we'll page
        max_bytes_per_process = 512e8 #256 MB #cap per core data sizes for sanity's sake (also helps ensure reasonable workload distribution)
        bytes_per_sample = 4

        max_allowed_data_size_per_core = min(max_bytes_per_process, available_memory_bytes/(4*self.opt.num_processes))  #prevent using up all the computer memory during operations on data

        self.max_samples_per_core = 2**int(np.log2(max_allowed_data_size_per_core / bytes_per_sample))

    def get_drf_metadata(self):
        """
        Pull in the metadata for the DRF fileso we can figure out size, slicing, 
        and such as well as computing the actual sample range in which to process data

        note that I just assume the channels are identical here if there are multiple
        """

        # convert channel argument to separate tuples for channels and subchannels
        self.channels, self.subchannels = zip(*self.opt.channels)
        # replace None subchannels with 0
        self.subchannels = tuple(
            0 if subch is None else subch for subch in self.subchannels
        )

        #read off properties we care about
        dio = drf.DigitalRFReader(self.opt.path)
        self.sr = dio.get_properties(self.channels[0])["samples_per_second"]

        self.bounds = dio.get_bounds(self.channels[0])

        self.dt_start = datetime.datetime.fromtimestamp(
            int(self.bounds[0] / self.sr),
            tz=datetime.timezone.utc,
        )
        self.dt_stop = datetime.datetime.fromtimestamp(
            int(self.bounds[1] / self.sr), tz=datetime.timezone.utc
        )


        metadata = dio.read_metadata(self.bounds[0],self.bounds[0]+1, self.channels[0])
        metadata_key = list(metadata.keys())[0]
        radio_metadata = metadata[metadata_key]

        self.cf = radio_metadata['center_frequencies'][0]

        print(
            f"data bound times {self.dt_start.isoformat()} to {self.dt_stop.isoformat()} UTC"
        )
        print(f"sample rate {self.sr} Hz")

        if self.opt.verbose:
            print("bound sample index {0}".format(self.bounds))

        #get actual processing bounds

        self.proc_bounds = np.array(self.bounds)

        if self.opt.start:
            dtst0 = dateutil.parser.parse(self.opt.start)
            st0 = (
                dtst0 - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
            ).total_seconds()
            self.proc_bounds[0] = int(st0 * self.sr)

        if self.opt.end:
            dtst0 = dateutil.parser.parse(self.opt.end)
            et0 = (
                dtst0 - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
            ).total_seconds()
            self.proc_bounds[1] = int(et0 * self.sr)


        if self.opt.verbose:
            print("processing samples within sample index {0}".format(self.proc_bounds))

        ### check for bad samples at start of data

        self.start_offsets = []

        for channel in self.channels:
            rfdata = dio.read_vector(self.proc_bounds[0], int(1.0*self.sr), channel, 0)
            nans = np.where(np.isnan(rfdata))[0]
            if len(nans) == 0:
                self.start_offsets.append(0)
                if self.opt.verbose:
                    print(f"channel {channel} has no dropped samples at start")
            else:
                self.start_offsets.append(nans[-1]+1)
                if self.opt.verbose:
                    print(f"channel {channel} has {nans[-1]} dropped samples at start")

    def get_data_avars(self, data, rate, stop_len):
        """
        function to get octave Allan vars for a chunk of data
        data should be a power of 2 length for the output data to be valid
        rate is sample rate of data in Hz
        stop_len is so that we can avoid fully decimating data on an initial 
        pass to have better efficiency for large decimations
        """

        taus = []
        avars = []
        avar_vars = []
        avar_samples = []

        while len(data) > stop_len: 
            if len(data) % 2 ==1:
                #force data to have an even number of samples
                #horribly inefficient but allows us to usefully handle non power of 2 length inputs
                #note however this will invalidat subsequent recombination of the output data
                data = data[:-1] 

            #compute and store allan var
            tau = 1/rate
            #print(f"computing Allan variance for tau = {tau} seconds")
            avar, avar_var, n_samps = self.estimate_allan_var(data, rate)
            taus.append(1/rate)
            avars.append(avar)
            avar_vars.append(avar_var)
            avar_samples.append(n_samps)

            #collapse data and compute rate for next octave
            data, rate = self.integrate_data(data, rate)

        taus = np.array(taus)
        avars = np.array(avars)
        avar_vars = np.array(avar_vars)
        avar_samples = np.array(avar_samples)

        return taus, avars, avar_vars, avar_samples, data, rate

    def handle_data_slice_avars(self, channel, subchannel, segment_length, start_index):
        """
        pull in a chunk of drf data and run computation for it
        return results and decimated data
        """
        if self.opt.verbose:
            print("handling data slice starting at sample {start_index}")
            
        dio = drf.DigitalRFReader(self.opt.path)
        data = dio.read_vector(start_index, segment_length, channel, subchannel) #import rf data segment
        data = np.power(np.abs(data),2) #convert to power
        rate = float(self.sr)

        taus, avars, avar_vars, avar_samples, data, rate = self.get_data_avars(data, rate, 4)

        return taus, avars, avar_vars, avar_samples, data, rate

    def integrate_data(self, data, rate):
        """
        data is a numpy array of data, must be an even length
        rate is sample rate of the data
        """

        new_data = np.nanmean(data.reshape((-1,2)),axis=1) #two rows with every other sample
        new_rate = rate/2.0

        return new_data, new_rate

    def estimate_allan_var(self, data, rate):
        """
        data is an array of the appropriately integrated allan variance samples (eg. \bar{y})
        rate is the corresponding sample rate of the data (note this is also 1/tau)
        """

        avar = 0.5 * np.nanmean(np.power(data[1::] - data[0:-1], 2)) #despite looking a bit funny this slicing is correct
        avar_var = 1/(len(data)-1) * avar

        return avar, avar_var, len(data)-1

    def compute_drf_avars(self):
        """
        multi threaded procesing for full drf recordings
        note we do lose a few samples at the slice edges but it's too complicated to deal with that for it to be worth it
        """

        #### process allan vars 
        
        channel_taus = []
        channel_avars = []
        channel_avar_vars = []
        channel_avar_samples = []

        for i in range(len(self.channels)):

            if self.opt.verbose:
                print(f"working on data from channel {self.channels[i]}")
            
            channel = self.channels[i]
            subchannel = self.subchannels[i]
            total_samples = self.proc_bounds[1] - self.proc_bounds[0] - self.start_offsets[i]

            #get segment length for data import
            segment_length = min(self.max_samples_per_core, 2**int(np.log2(total_samples/self.opt.num_processes)+1))
            
            num_segments = int(total_samples/segment_length) #note we lose the trailing edge of the data here but whatever

            start_indices = np.arange(self.proc_bounds[0]+self.start_offsets[i], self.proc_bounds[1], segment_length)[:num_segments] #chop off the last partial segment

            #pool = multiprocessing.Pool()
            pool = multiprocessing.Pool(processes=self.opt.num_processes)
            avar_slice = partial(self.handle_data_slice_avars, channel, subchannel, segment_length)

            if self.opt.verbose:
                print("starting multithreaded process")

            outputs = pool.map(avar_slice, start_indices)

            pool.close()
            pool.join()

            taus, avars, avar_vars, avar_samples, data, rate = outputs[1]
            
            for i in range(1,len(start_indices)):
                avars += outputs[i][1]
                avar_vars += outputs[i][2]
                avar_samples += outputs[i][3]
                data = np.append(data, outputs[i][4])

            avars = avars / len(start_indices)
            avar_vars = avar_vars / len(start_indices)

            if self.opt.verbose:
                print(f"operating on final data for taus greater than {taus[-1]} seconds")

            #### finally operate on remaining data

            if len(data) > 1:
            
                new_taus, new_avars, new_avar_vars, new_avar_samples, data, rate = self.get_data_avars(data, rate, 1)

                taus = np.append(taus, new_taus)
                avars = np.append(avars, new_avars)
                avar_vars = np.append(avar_vars, new_avar_vars)
                avar_samples = np.append(avar_samples, new_avar_samples)

            channel_taus.append(taus)
            channel_avars.append(avars)
            channel_avar_vars.append(avar_vars)
            channel_avar_samples.append(avar_samples)

        return channel_taus, channel_avars, channel_avar_vars, channel_avar_samples



    def process_avar_plots(self):
        """
        handle processing for Allan variance plots
        """

        #start by getting the important info about what sample rates we're dealing with, etc
        self.get_drf_metadata()
        #process the allan variances
        channel_taus, channel_avars, channel_avar_vars, channel_avar_samples = self.compute_drf_avars()


        
        plt.rcParams['figure.figsize'] = [8, 6]

        plt.figure()

        if self.opt.title:
            plt.title(self.opt.title, fontsize=14)
        else:
            filename = self.opt.path.split('/')[-1]
            plt.title(f"Allan variance for DRF recording {filename}", fontsize=14)

        plt.ylabel(r"$ \sigma_y \left( \tau \right)$  $\left[\frac{V^2}{s}\right]$",fontsize=13)
        plt.yticks(fontsize=12)
        plt.xlabel(r"$ \tau $  $\left[s\right]$",fontsize=13)
        plt.xticks(fontsize=12)

        for i in range(len(self.channels)):
            
            #plt.errorbar(taus, np.sqrt(avars), yerr=np.sqrt(avar_vars))
            plt.errorbar(channel_taus[i], np.sqrt(channel_avars[i]),yerr=np.sqrt(channel_avar_vars[i]), label=f"{self.channels[i]}", capsize=3, capthick=1)

        plt.yscale('log')
        plt.xscale('log')
        plt.xlim([np.min(channel_taus), np.max(channel_taus)])
        tracemin = np.min(np.sqrt(channel_avars))
        tracemax = np.max(np.sqrt(channel_avars))
        plt.ylim([10**int(np.log10(tracemin)), 10**int(np.log10(tracemax)+1)])

        plt.legend(fontsize=13)
        plt.grid()
        plt.tight_layout()

        if self.opt.outname:
            plt.savefig(self.opt.outname, dpi=300)

        if self.opt.appear or not self.opt.outname:
            plt.show()



###################################################
# command line stuff
###################################################



def intinttuple(s):
    """Get (int,int) tuple from int:int strings."""
    parts = [p.strip() for p in s.split(":", 1)]
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    else:
        return None

def floatinttuple(s):
    """Get (int,int) tuple from int:int strings."""
    parts = [p.strip() for p in s.split(":", 1)]
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    else:
        return None


def strinttuple(s):
    """Get (string,int) tuple from str:int strings."""
    parts = [p.strip() for p in s.split(":", 1)]
    if len(parts) == 2:
        return parts[0], int(parts[1])
    else:
        return parts[0], None


class Extend(argparse.Action):
    """Action to split comma-separated arguments and add to a list."""

    def __init__(self, option_strings, dest, type=None, **kwargs):
        if type is not None:
            itemtype = type
        else:

            def itemtype(s):
                return s

        def split_string_and_cast(s):
            return [itemtype(a.strip()) for a in s.strip().split(",")]

        super(Extend, self).__init__(
            option_strings, dest, type=split_string_and_cast, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        cur_list = getattr(namespace, self.dest, [])
        if cur_list is None:
            cur_list = []
        cur_list.extend(values)
        setattr(namespace, self.dest, cur_list)

def parse_command_line():
    scriptname = os.path.basename(sys.argv[0])

    formatter = argparse.RawDescriptionHelpFormatter(scriptname)
    width = formatter._width

    title = "drf_avar"
    copyright = "Copyright (c) 2025 Massachusetts Institute of Technology"
    shortdesc = "Telescope noise amplitude Allan variance calculator for DigitalRF format."
    desc = "\n".join(
        (
            "*" * width,
            "*{0:^{1}}*".format(title, width - 2),
            "*{0:^{1}}*".format(copyright, width - 2),
            "*{0:^{1}}*".format("", width - 2),
            "*{0:^{1}}*".format(shortdesc, width - 2),
            "*" * width,
        )
    )

    parser = argparse.ArgumentParser(
        description=desc,
        prefix_chars="-",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        metavar="datadir_path",
        help="Path to data directory which has the DigitalRF channel subdirectories.",
    )
    parser.add_argument(
        "-t",
        "--title",
        dest="title",
        default=None,
        help="Use title provided for the plot.",
    )
    parser.add_argument(
        "-s",
        "--start",
        dest="start",
        default=None,
        help=(
            "Use the provided start time instead of the first time in the data."
            " format is ISO8601: 2015-11-01T15:24:00Z"
        ),
    )
    parser.add_argument(
        "-e",
        "--end",
        dest="end",
        default=None,
        help=(
            "Use the provided end time for the plot."
            " format is ISO8601: 2015-11-01T15:24:00Z"
        ),
    )
    parser.add_argument(
        "-c",
        "--channels",
        dest="channels",
        action=Extend,
        type=strinttuple,
        default=[],
        help="""Input channel specification, including names and mapping from
                receiver channels.  Specifications are given as a receiver
                channel name and sub-channel pair, e.g. "ch0:0". The number and
                colon are optional; if omitted, the receive sub-channel is zero.
                """,
    )
    parser.add_argument(
        "-P",
        "--processes",
        dest="num_processes",
        default=1,
        type=int,
        help="""Number of processes to use for computing the adev 
                If omitted defaults to 1 (single threaded).
                setting processes to 0 will default to using a number of 
                processes equal to the number of available cpu cores""",
    )
    parser.add_argument(
        "-o",
        "--outname",
        dest="outname",
        default=None,
        type=str,
        help="Name of file that figure will be saved under.",
    )
    parser.add_argument(
        "-a",
        "--appear",
        action="store_true",
        dest="appear",
        default=False,
        help="Makes the plot appear through pyplot show.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Print status messages to stdout.",
    )

    options = parser.parse_args()

    return options

#
# MAIN PROGRAM
#

# Setup Defaults
if __name__ == "__main__":
    """
    Needed to add main function to use outside functions outside of module.
    """

    # Parse the Command Line for configuration
    options = parse_command_line()

    if options.path is None:
        print("Please provide an input source with the -p option!")
        sys.exit(1)

    if options.verbose:
        print("options: {0}".format(options))

    # Activate the AllenVarProcessor
    avar_processor = AllenVarProcessor(options)

    avar_processor.process_avar_plots()