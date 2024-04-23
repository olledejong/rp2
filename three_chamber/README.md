# Analysis of the three-chamber experiments 

The analysis of the three-chamber experiments can be split into two distinct parts, the behavioural analysis, and the
electrophysiological/EEG analysis. 

The former is done using a notebook for both the [sociability](/three_chamber/sociability/behavioural_analysis.ipynb)
experiment as well as the [social preference](/three_chamber/social_preference/behavioural_analysis.ipynb) experiment. 
See [Behavioural analysis](#behavioural-analysis).

The time-locked EEG analysis is performed by first preprocessing the data of both the 3-chamber sociability and 
the preference experiments, which can be performed using exactly the same files. Only the actual power and connectivity
analysis is performed using different notebooks. See [Electrophysiological analysis](#electrophysiological-analysis).

## Behavioural analysis

The main output of this analysis is a violin plot of sociability/social preference metrics between
DRD2-WT and DRD2-KO mice. As mentioned, there's a notebook for both the [sociability](/three_chamber/sociability/behavioural_analysis.ipynb)
experiment and the [social preference](/three_chamber/social_preference/behavioural_analysis.ipynb) experiment.

The notebook uses Tkinter (folder select dialogs) to retrieve the paths it needs to do its thing. Just work your
way through the notebook and you should be set.

## Electroencephalogram (EEG) analysis

Brain activity has been recorded over the course of the experiments. To analyse this, we first need to preprocess the raw
EEG data and create frame-based epochs (i.e. epochs based on the behavioural data). E.g. a social cup interaction starts
and ends at a certain frame of the video (BORIS data). We use those frame time-stamps to calculate the EEG sample so we
can retrieve the EEG data that belongs to the interaction.

Once we've created social/non-social or novel/familiar interaction epochs, we can proceed to the actual power and/or 
connectivity analysis (see [Statistical Analysis](#statistical-analysis)).

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

#### 1. Create a NWB file for each animal

First we create a NWB file per animal. This NWB file will hold the EEG data that belongs to the subject, as well as the 
data about the subject, the electrodes, the experiment etc.

Use the [NWB creation script](/shared/create_nwb_files.py) to do this. Again, it is important that the file-names of your EDF
files are structured like this: **TAINI_$TransmID_$SubID_$ALIAS_%Y-%m-%d_%H-%M-%S_$SesID_$INC.edf**

The script will ask you to enter a experiment identifier, so provide a string like 'resting_state', 'olfactory', or 
'three_chamber_sociability'.

Then, the script asks you to select the directory that hold the EDF files for this experiment. It will also ask you to
select or create a directory to save the NWB files to. Lastly, it will ask for the location of the metadata file that
holds information on all animals.

Once these paths have been provided, a NWB file will be created and written to the selected output directory for each animal.

#### 2. Extract information to align the EEG and video/behavioural data

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

#### 3. Create an epoch file per animal through frame based epoching

Once the NWB files have been created, and we have the TTL/LED onset events, we can create some epochs based on the 
behavioural data files from BORIS.

Use the [create_frame_based_epochs.py](create_frame_based_epochs.py) script for this. 

The script will ask you to select the NWB folder, the folder that holds the behavioural data (one file per animal), and 
the folder that holds the video analysis output (ROI Excel, pickle file). Lastly, it allows you to select or crreate a
folder the epoch files will be saved to.

The epoch files are saved in the last mentioned folder.

### Statistical Analysis

TODO