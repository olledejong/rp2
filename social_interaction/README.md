# Analysis of the social-interaction experiment

The analysis of the social interaction test can be split into two distinct parts, the behavioural analysis, and the
electrophysiological/EEG analysis.

We use Excel files holding the tracked behaviour for the behavioural analysis. These files were obtained by
by using the BORIS tracking software. This manual behaviour scoring was performed by bachelor students Lotte and Rebeka.

The time-locked EEG analysis is performed by first preprocessing the raw EEG data.

## Behavioural analysis

## Electroencephalogram (EEG) analysis

### Prerequisites

Before we can start, we have to be sure we have all we need. Please confirm whether you do:

- An Excel file holding the metadata on all animals used for the DRD2 experiments
- A folder holding an EDF file (holds EEG data) for each animal

Please make sure the filenames are structured like this: 

**TAINI_$TransmID_$SubID_$ALIAS_%Y-%m-%d_%H-%M-%S_$SesID_$INC.edf**

Where **$TransmID** is the id of the transmitter used for this animal during this experiment, **$SubID** is the unique
animal identifier, the alias is the name the animal was given (aligns with batch/cage info) (e.g. 4.15), **%Y-%m-%d** is
the date, **%H-%M-%S** is the time, and **$SesID** is the session id. **$INC** is irrelevant.

- A folder holding a behaviour Excel file (exported from BORIS) for each animal

### Pre-processing the raw data

Before we can do any kind of statistical analysis, may it be power or connectivity analysis, the data needs to be 
pre-processed.

#### I. Create a NWB file for each animal

First we create a NWB file per animal. This NWB file will hold the EEG data that belongs to the subject, as well as the 
data about the subject, the electrodes, the experiment etc.

Use the [NWB creation script](/shared/create_nwb_files.py) to do this. Again, it is important that the file-names of your EDF
files are structured like this: **TAINI_$TransmID_$SubID_$ALIAS_%Y-%m-%d_%H-%M-%S_$SesID_$INC.edf**

The script will ask you to enter a experiment identifier, so enter something like: 'social_interaction'.

Then, the script asks you to select the directory that hold the EDF files for this experiment. It will also ask you to
select or create a directory to save the NWB files to. Lastly, it will ask for the location of the metadata file that
holds information on all animals.

Once these paths have been provided, a NWB file will be created and written to the selected output directory for each animal.

#### II. Extract information to align the EEG and video/behavioural data

As we wish to create epochs based on behavioural data, we need to align the EEG data and the video.

We use TTL states to do achieve this. These TTL onsets are recorded both in the video and the EEG. In the videos the TTL
onsets are visualized by a flashing LED. In order to extract that data from the videos we first have to select where the
LEDs are located in each experiment video. We do this using the [identify_led_rois.py](/shared/video_ttl_extraction/identify_led_rois.py)
script.

Provide the folder that the videos are located in, and the folder you wish to save the ROI Excel file to, and select a ROI
for each video that pops up on your screen, confirming the selection with enter.

Then, we can use the LED ROIs to extract the frames of the LED onsets from the videos using the 
[ttl_onset_extractor.py](/shared/video_ttl_extraction/ttl_onset_extractor.py) script. 

It will ask you to provide the folder you saved the ROI Excel file to, as well as the folder that holds the experiment
videos. It gets the TTL/LED onsets for each video and stores these in a pickle file in the folder the ROI Excel file is
in. 

#### III. Create an epoch file per animal through frame based epoching

Once the NWB files have been created, and we have the TTL/LED onset events, we can create some epochs based on the 
behavioural data files from BORIS.

Use the [create_frame_based_epochs.py](../shared/create_frame_based_epochs.py) script for this. 

The script will ask you to select the NWB folder, the folder that holds the behavioural data (one file per animal), and 
the folder that holds the video analysis output (ROI Excel, pickle file). Lastly, it allows you to select or create a
folder the epoch files will be saved to.

We argued that, in contrary to the 3-chamber experiments, short interactions might also elicit interesting brain 
activity, as sniffing and such direct interactions may be very brief. Hence, we set the *min_event_duration* in 
[settings_general.py](/settings_general.py) to None.

Furthermore, the length of the created epochs is determined by the setting *desired_epoch_length* in [settings_general.py](/settings_general.py).

Lastly, you can indicate in the same settings file whether you want epochs to overlap using *overlap_epochs* and
*epoch_overlap_cutoff*. 

Example: Imagine if you have an epoch of 1.4 seconds, and that your settings are:
- *min_event_duration* is 1 (second)
- *desired_epoch_length* is 1 (second)
- *overlap_epochs* is True
- *epoch_overlap_cutoff* is 0.5 (fraction, equals 50%)

Then, the [create_frame_based_epochs.py](../shared/create_frame_based_epochs.py) script would create an overlap of 0.6 seconds, which in theory yields an
epoch of 0.0s -> 1.0s, and an epoch of 0.4s -> 1.4s. However, this overlap exceeds the maximum epoch overlap (given 
by *epoch_overlap_cutoff*). Hence, this interaction yields one epoch of one second ( 0.0 --> 1.0s ), the rest (0.4s) is
discarded.
The epoch files are saved in the last mentioned folder.

### Statistical Analysis

Now we have cleaned epochs, we can proceed to do some analysis on it. The following files can be used to do exactly that.

[eeg_power_analysis.py](/social_interaction/eeg_power_analysis.ipynb) & [connectivity_analysis.py](/social_interaction/connectivity_analysis.ipynb)
