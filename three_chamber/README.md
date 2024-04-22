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
- A folder holding a behaviour Excel file (exported from BORIS) for each animal

### Pre-processing the raw data

Before we can do any kind of statistical analysis, may it be power or connectivity analysis, the data needs to be 
pre-processed.

#### 1. Create the experiment/EDF metadata file

First, we need to create an Excel file holding metadata on the animals included in the experiment. This is used
for which we thus need to run the analysis.

#### 2. Create a NWB file for each animal

#### 3. Create an epoch file per animal through frame based epoching

### Statistical Analysis

TODO