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

        self.decimation = 100 #effectively sets max integration for a given maximum tau
        self.max_data_chunk_size = 1e8 #* self.opt.num_processes #do not pull in more than about 100MB per process (seems reasonable)
        self.max_data_chunk_length = self.max_data_chunk_size / 4 #4 bytes per sample in sane formats

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
        self.dio = drf.DigitalRFReader(self.opt.path)
        self.sr = self.dio.get_properties(self.channels[0])["samples_per_second"]

        self.bounds = self.dio.get_bounds(self.channels[0])

        self.dt_start = datetime.datetime.fromtimestamp(
            int(self.bounds[0] / self.sr),
            tz=datetime.timezone.utc,
        )
        self.dt_stop = datetime.datetime.fromtimestamp(
            int(self.bounds[1] / self.sr), tz=datetime.timezone.utc
        )


        metadata = self.dio.read_metadata(self.bounds[0],self.bounds[0]+1, self.channels[0])
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

        self.proc_bounds = self.bounds

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


    def get_tau_values(self):
        """
        get tau values that actually work with the input data
        and make sure they are compatible wit reasonable slicing
        """

        data_len = self.proc_bounds[1] - self.proc_bounds[0]
        data_len_seconds = data_len/self.sr 

        #get bounds for tau
        if self.opt.taubounds is not None:
            tau_min = self.opt.taubounds[0]
            tau_max = min(self.opt.taubounds[1], data_len_seconds/2.1) #clip to where it will still be computable
        else:
            #generate a set of taus based on the data
            
            tau_max = data_len_seconds / 3 #use about a third of the samples for actually doing things
            if self.opt.tauscale.lower() == 'log10': 
                tau_min = tau_max * 1e-5
            else: #linear
                tau_min = tau_max * 1e-3 #1000 samples


        #determine base integration time which will be used to generate all samples of tau
        tmin = tau_min / 10.0 #set to 1 tenth of taumin so that our processing will actually be reasonably 

        #get the true time for tmin compatible with the specified fft slicing
        fft_rate = self.sr / self.opt.fft_bins 

        fft_slices = max(int(tmin * fft_rate), 1)

        n_min = int(fft_slices * self.opt.fft_bins) #minimum number of samples to a slice

        if self.opt.tauscale.lower() == 'log': 
            #approximately log10 scaling give or take a bit
            num_taus = int(10*(np.log10(tau_max) - np.log10(tau_min)))
            approx_taus = np.logspace(np.log10(tau_min), np.log10(tau_max), num=num_taus, base=10)

            #get corresponding sample strides through the data 
            #note this is divided by self.opt.fft_bins to fet the strid through the transformed data
            true_ns = np.unique(np.array([ int((atau * self.sr)/n_min)*n_min for atau in approx_taus]))
            true_taus = true_ns / float(self.sr)

        else: #linear
            max_samples = 1000
            num_taus = int(min(data_len_seconds/tau_min, max_samples))
            approx_taus = np.linspace(tau_min, tau_max, num=num_taus)

            #get corresponding sample strides through the data 
            #note this is divided by self.opt.fft_bins to fet the strid through the transformed data
            true_ns = np.unique(np.array([ int((atau * self.sr)/n_min)*n_min for atau in approx_taus]))
            true_taus = true_ns / float(self.sr)

        if data_len > self.max_data_chunk_length: #need to import and process multiple data segments
            #self.iterative_computation = True

            #to make data chunking well behvaed we need to change some of the tau values after computing the data slicing

            #figure out what max tau computed from a single data segment can be to be able to usefully stride 
            #through x[2n::] -2x[n::] + x[::] need to have 2n correspond to less than half the data then we 
            #will overlap the data for actually computing all of this 

            #factor of 2 in the max chunk size is because I need to half overlap it 

            self.maximum_drf_data_size = int(self.max_data_chunk_length/(2 * n_min * self.decimation)) * 2 * n_min * self.decimation

            max_single_pass_tau_index = np.where(4 * true_ns > self.maximum_drf_data_size)[0][0]-1
            if self.opt.verbose:
                print(f"maximum tau for single data slice = {true_taus[max_single_pass_tau_index]} seconds")

            #taus beyond the point the data is decimted to need to change

            upper_ns = np.array([int(true_n / (n_min * self.decimation)) * (n_min * self.decimation) for true_n in true_ns[max_single_pass_tau_index:-1]])
            upper_taus = upper_ns / float(self.sr)

            #overwrite original arrays 

            true_ns[max_single_pass_tau_index:-1] = upper_ns
            true_taus[max_single_pass_tau_index:-1] = upper_taus

            self.tau_pivot_index = max_single_pass_tau_index

        else: #we can comfortably do this in a single pass
            #self.iterative_computation = False
            self.maximum_drf_data_size = int(data_len/(2*n_min* self.decimation)) *2 * n_min * self.decimation #ensure that data aligns acceptably to our fft slices
            self.tau_pivot_index = len(true_taus)



        self.minimum_drf_data_size = n_min
        self.tau_sample_strides = true_ns
        self.tau_values = true_taus
        

        if self.opt.verbose:
            print(f"computing Allan Variance for {len(true_ns)} values of tau")
            print(f"actual minimum tau = {true_taus[0]} s")
            print(f"actual maximum tau = {true_taus[-1]} s")


    def calculate_welch_slice(self, channel, subchannel, indices):
        '''
        Calculate the welch method spectrogram for the given data slice and return the spectrogram
        '''

        #data = self.rfdata[start_index, start_index+self.minimum_drf_data_size]
        data = self.dio.read_vector(indices[0], indices[1], channel, subchannel)
        data = np.reshape(data,[self.minimum_drf_data_size,-1])
        #welch operation

        try:
            freq_axis, psd_data = scipy.signal.welch(
                data,
                fs=float(self.sr),
                nperseg=self.opt.fft_bins,
                detrend=False,
                scaling="density",
                return_onesided=False,
                average='mean',
                axis=0
            )
        except Exception:
            traceback.print_exc(file=sys.stdout)

        return np.real(np.abs(scipy.fft.fftshift(psd_data, axes=0))), scipy.fft.fftshift(freq_axis,axes=0)

    def get_variance_estimantes(self, psd_data, last_segment, sample_stride):
        """
        Create an estimate for the Allan variance 
        sample stride is tau*samp_rate
        """

        sr_effective = float(self.sr / self.minimum_drf_data_size)
        samp_stride_effective = int(sample_stride / self.minimum_drf_data_size)

        x2 = psd_data[:,2*samp_stride_effective::1]
        x1 = psd_data[:,samp_stride_effective::1]
        x0  = psd_data[::1]
        

        if last_segment: #exhaust the available data if this is the last slice
            num_samps = np.shape(x2)[1]
            #num_samps = np.shape(x1)[1]
        else:
            num_samps = min(int(np.shape(psd_data)[1]/2),np.shape(x2)[1])  #use exactly half the data to line up with my overlap estimates
            #num_samps = min(int(np.shape(psd_data)[1]/2),np.shape(x1)[1])  #use exactly half the data to line up with my overlap estimates

        avars = np.nanmean(np.power(x2[:,:num_samps] - 2*x1[:,:num_samps] + x0[:,:num_samps], 2)  ,axis=1) / (samp_stride_effective/sr_effective)**2
        #avars = np.nanmean(np.power(x1[:,:num_samps] - x0[:,:num_samps], 2) / samp_stride_effective,axis=1)
        # #avars = np.nanmedian(np.power(x2[:,:num_samps] - 2*x1[:,:num_samps] + x0[:,:num_samps], 2) / (2.0 *(samp_stride_effective )**2),axis=1)
        avar_vars = avars / (2*(num_samps-1))
        return num_samps, avars, avar_vars


    def process_data_segment(self, channel, subchannel, start_sample, segment_length, last_segment):
        """
        process a given segment of data and return the variances and a downsampled portion of the spectrograms
        """
        if self.opt.verbose:
            print(f"working on data starting at sample {start_sample} and ending at {segment_length+start_sample}")

        #holding_variables for allan deviation calculations

        short_taus = self.tau_values[0:self.tau_pivot_index]
        short_ns = self.tau_sample_strides[0:self.tau_pivot_index]
        short_tau_num_samples = np.zeros(self.tau_pivot_index, np.int64)
        short_tau_allan_vars = np.zeros((self.tau_pivot_index,self.opt.fft_bins), np.float64) #storage variable for allen variance calculations

        ######################################################
        # multithreaded data access + transform
        ######################################################

        if self.opt.num_processes == 0:
            num_cores = multiprocessing.cpu_count()
        else:
            num_cores = np.minimum(multiprocessing.cpu_count(), self.opt.num_processes)

        print(f"Using {num_cores} threads for welch calculation")

        slice_len = int(segment_length/(num_cores *self.minimum_drf_data_size)+1)*self.minimum_drf_data_size
        
        slice_starts = np.arange(start_sample, start_sample + segment_length, slice_len)
        slice_stops = slice_starts[1::]
        slice_stops=np.append(slice_stops, start_sample + segment_length)
        slice_lens = slice_stops-slice_starts
        
        indices =[(start, length) for (start,length) in zip(slice_starts,slice_lens)]
        

        psd_data = np.zeros([self.opt.fft_bins, int(segment_length/self.minimum_drf_data_size)], np.float64)


        pool = multiprocessing.Pool()
        pool = multiprocessing.Pool(processes=num_cores)

        welch = partial(self.calculate_welch_slice, channel, subchannel)

        outputs = pool.map(welch, indices)

        pool.close()
        pool.join()

        for b in np.arange(len(slice_starts), dtype=np.int_):
            #psd_data = outputs[b][0]
            psd_data[:, int((slice_starts[b]-start_sample)/self.minimum_drf_data_size):int((slice_stops[b]-start_sample)/self.minimum_drf_data_size)] = outputs[b][0]
            freq_axis = outputs[b][1]


        ###############################################################################
        # Handle computation of Allan variances that can be computed within the segment
        ###############################################################################

        if self.opt.verbose:
            print(f"working on Allan Variance Calculation with {num_cores} threads")

        pool = multiprocessing.Pool()
        pool = multiprocessing.Pool(processes=num_cores)

        self.psd_data_slice = psd_data

        avars = partial(self.get_variance_estimantes, psd_data, last_segment)
        outputs = pool.map(avars, short_ns)

        pool.close()
        pool.join()

        for i in range(len(short_taus)):
            short_tau_num_samples[i] = outputs[i][0]
            short_tau_allan_vars[i] = outputs[i][1] 
            avar_vars = outputs[i][2]


        ###############################################################################
        # Decimate spectrograms so they can be accumulated for long timescales
        ###############################################################################
        try:
            if last_segment:
                psd_len = len(np.reshape(psd_data,(-1,1)))
                truncated_len = int(psd_len/(self.opt.fft_bins*self.decimation)) *(self.opt.fft_bins*self.decimation)

                if truncated_len >0:
                    reshaped_psds = np.reshape(np.reshape(psd_data,(-1,1))[:truncated_len],(self.opt.fft_bins,self.decimation,-1))
                    decimated_psds = np.nanmean(reshaped_psds,axis=1)
                else:
                    decimated_psds = np.empty((self.opt.fft_bins,)).fill(np.nan) #just to have something to return that isn't empty
            else:
                reshaped_psds = np.reshape(psd_data,(self.opt.fft_bins,self.decimation,-1))
                decimated_psds = np.nanmean(reshaped_psds,axis=1)
        except:
            decimated_psds = np.empty((self.opt.fft_bins,)).fill(np.nan)

        
        return short_tau_num_samples, short_tau_allan_vars, decimated_psds, freq_axis





    def process_channel_avars(self, channel, subchannel):
        """
        process data from a given input channel and return Allan Variances
        channel= data channel to process
        tau_pivot_index = index of lagest tau computable from a single data chunk

        return 
        """

        #create storage variables

        self.allan_var_totals = np.zeros((len(self.tau_values),self.opt.fft_bins), np.float64) #storage variable for allen variance calculations
        self.allan_var_num_samples = np.zeros((len(self.tau_values)), np.int64) #storage variable for number of accumulated_samples

        self.decimated_psd = np.empty((self.opt.fft_bins,1), dtype=np.float64)


        segment_starting_index = self.proc_bounds[0]
        #########################################
        #processing loop for initial drf handling
        #########################################

        while segment_starting_index < self.proc_bounds[1]:
            #check actual_bounds on the segment

            if segment_starting_index <= self.proc_bounds[1] - self.maximum_drf_data_size:
                last_segment = False
                segment_len = self.maximum_drf_data_size

            else: #last data slice we'll be handling
                last_segment = True
                segment_len = int((self.proc_bounds[1]-segment_starting_index)/(2*self.minimum_drf_data_size))*2*self.minimum_drf_data_size
                if segment_len <=0:
                    break


            num_samps, avars, decimated_psds, freq_axis = self.process_data_segment(channel, subchannel, segment_starting_index, segment_len, last_segment)

            avars[np.isnan(avars)] = 0

            self.allan_var_totals[:self.tau_pivot_index,:] += np.transpose(np.array([avars[:,i] * num_samps for i in range(np.shape(avars)[1])]))
            self.allan_var_num_samples[:self.tau_pivot_index] += num_samps

            self.decimated_psd = np.append(self.decimated_psd, decimated_psds, axis=1)

            segment_starting_index += int(self.maximum_drf_data_size/2)

        ##########################################
        # process remaining tau values
        ##########################################




        allan_vars = np.transpose(np.array([self.allan_var_totals[:,i] / self.allan_var_num_samples for i in range(np.shape(avars)[1])]))
        
        return allan_vars, self.tau_values, self.allan_var_num_samples




    def process_avar_plots(self):
        """
        handle processing for Allan variance plots
        """

        #start by getting the important info about what sample rates we're dealing with, etc
        self.get_drf_metadata()

        #figure out what the parameters for which we want to compute the allan variance are
        self.get_tau_values()

        #process the allan variances

        for channel, subchannel in zip(self.channels, self.subchannels):
            allan_vars, taus, num_samples =self.process_channel_avars(channel, subchannel)

            plt.figure()
            taulen = np.where(num_samples==0)[0][0]-1
            for i in range(self.opt.fft_bins):
                plt.loglog(taus[:taulen],np.sqrt(allan_vars[:taulen,i]))
            plt.grid()
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
        default="Digital RF Data",
        help="Use title provided for the plot.",
    )
    parser.add_argument(
        "-T",
        "--taus",
        dest="taubounds",
        type=floatinttuple,
        default=None,
        metavar="TLOW:THIGH",
        help=(
            """min and max tau values in seconds, eg -T '1e-3:50.0'.
            if not provided reasonable numbers will be chosen based on the dataset"""
            
        ),
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
        "--channel",
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
        "-b",
        "--fft_bins",
        dest="fft_bins",
        default=32,
        type=int,
        help="The number of separate frequency bins in which to compute the Allan variance",
    )
    parser.add_argument(
        "--tauscale",
        dest="tauscale",
        default="log10",
        help="""x scaling for tau computation and plot: 'lin 'or 'log' (default: log)
                If this is set to linear, the maximum number of samples is capped at 10,000
                and thus the minimum tau cannot be less than taumax/1e4. this will override a provided tau range"""
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