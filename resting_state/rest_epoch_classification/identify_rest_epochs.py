"""
This file is a script version of the 'identify_rest_epochs' notebook. It clusters the non-movement
epochs for each subject, creates a figure that gives neat insights into the characteristics of each
cluster, and saves the cluster annotation to the metadata of the epochs object of the subject.

Then, the researcher can visually inspect the clusters and determine which are sleep, which are active,
and which are resting-state epochs. Based on this classification, the resting-state cluster can be further
analyzed by loading the saved epochs object that now includes the cluster annotations.
"""
import os
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from mne.time_frequency import psd_array_multitaper

# imports from other files
from settings import paths, freq_bands_eeg, freq_bands_emg, quality_emg

palette = sns.color_palette("husl", 3)


def calculate_emg_psd_features(signal, sfreq):
    """
    Function that derives (PSD) features from the EMG channel of the subject.

    :param signal:
    :param sfreq:
    :return:
    """
    psd, freq = psd_array_multitaper(signal, fmin=0, fmax=100, sfreq=sfreq, n_jobs=-1, verbose=False)

    emg_psds = {}  # slice psd data of epoch based on the defined bands
    for band, (start, end) in freq_bands_emg.items():
        slice_start, slice_end = int(start / 100 * len(freq)), int(end / 100 * len(freq))
        psd_slice = psd[slice_start:slice_end]

        emg_psds[f"EMG {band}"] = np.mean(np.log(psd_slice))

    emg_psds['EMG high-low freq ratio'] = ((emg_psds['EMG band9'] + emg_psds['EMG band10']) /
                                           (emg_psds['EMG band1'] + emg_psds['EMG band2']))

    return emg_psds


def calculate_eeg_psd_features(signal, sfreq):
    """
    Function that derives (PSD) features from a EEG channel of the subject.

    :param signal:
    :param sfreq:
    :return:
    """
    eeg_psds = {}
    for band, (start, end) in freq_bands_eeg.items():
        psd, freq = psd_array_multitaper(signal, fmin=start, fmax=end, sfreq=sfreq, n_jobs=-1)

        eeg_psds[f"OFC_L {band}"] = np.mean(np.log(psd))

    eeg_psds['gamma-delta ratio'] = (eeg_psds['OFC_L $\\gamma$'] / eeg_psds['OFC_L $\\delta$'])
    return eeg_psds


def save_radar_cluster_plot(features, df_plot, subject_id):
    """
    Save a grid that includes a radar plot for each cluster that describes feature importance.
    Fourth subplot is the cluster scatterplot.

    :param features: the already min-max scaled features for the epochs of this subject
    :param subject_id: the subject's id
    :param df_plot: a df holding the numerical data but also PCA components and cluster label
    :return:
    """
    clusters = np.unique(df_plot["cluster"])
    row_len = int(np.ceil(len(clusters) / 2)) + 1
    col_len = int(np.ceil(len(clusters) / row_len)) + 1
    palette = sns.color_palette("husl", 3).as_hex()

    # CREATE PLOTTING GRID USING PLOTLY

    fig = make_subplots(
        rows=row_len, cols=col_len,
        specs=[[{'type': 'polar'}, {'type': 'polar'}], [{'type': 'polar'}, {'type': 'scatter'}],
               [{"colspan": 2}, None]],
        horizontal_spacing=0.08, vertical_spacing=0.12,
        shared_yaxes=True, shared_xaxes=True,
        subplot_titles=["Cluster 0", "Cluster 1", "Cluster 2", "Clusters by PCA components, non-movement epochs",
                        "Boxplot of (min-max scaled) feature distribution per cluster"]
    )

    # LOOP THROUGH CLUSTERS AND GENERATE RADAR PLOTS AND BOX-PLOTS

    max_value = 0
    fig_box = go.Figure()  # for storing boxplot per cluster

    for i, cluster in enumerate(clusters):
        cluster_features = features[df_plot["cluster"] == cluster]
        cluster_features = cluster_features.drop(columns=['gamma-delta ratio', 'EMG high-low freq ratio'])

        row, col = (i // 2) + 1, (i % 2) + 1

        average_features = cluster_features.mean(axis=0)

        max_value = np.max(average_features) if np.max(average_features) > max_value else max_value

        fig.add_trace(go.Scatterpolar(
            r=average_features,
            theta=average_features.index,
            fill='toself',
            name=f'Cluster {i}',
            showlegend=False
        ), row=row, col=col)

        # add boxplot trace for this cluster as well
        cluster_features = cluster_features.drop(
            columns=['EMG band3', 'EMG band4', 'EMG band5', 'EMG band6', 'EMG band7', 'EMG band8'])
        repeated_arrays = [np.repeat(feature, len(cluster_features)) for feature in cluster_features.columns]
        x = np.concatenate(repeated_arrays)
        y = cluster_features.T.stack().values

        # per add_trace alle data van 1 cluster en splits het dmv de x op features
        fig_box.add_trace(go.Violin(
            y=y,
            x=x,
            name=f'Cluster {cluster}',
            marker_color=palette[cluster],
            showlegend=True
        ))

    fig_box.update_traces(box_visible=True, meanline_visible=True)
    # add the boxplot to the grid plot
    for trace in fig_box.data:
        fig.add_trace(trace, row=3, col=1)

    # ADD CLUSTER PLOT TO ROW 2 COL 2 OF GRID

    fig.add_trace(go.Scatter(
        x=df_plot[0], y=df_plot[1], mode="markers",
        marker=dict(color=df_plot["cluster"], colorscale=palette),
        showlegend=False
    ), row=2, col=2)
    fig.update_xaxes(title_text="Principle Component 1", row=2, col=2)
    fig.update_yaxes(title_text="Principle Component 2", row=2, col=2)

    # COMPLETE FIGURE
    fig.update_layout(
        height=400*row_len, width=620*col_len,
        margin=dict(l=50, r=50, t=70, b=70),
        colorway=palette,
        showlegend=True,
        violinmode='group',
        violingap=0.3,
        violingroupgap=0.1
    )
    fig.update_annotations(yshift=20)
    fig.update_polars(radialaxis=dict(range=[0, max_value + .05]))
    fig.write_image(
        os.path.join(paths['plots_folder'], f'ploss_thresh_500/non_mov_clustering/{subject_id}.png'))


def tag_outliers(df_numeric, df_plot):
    """
    Removes outliers from the epoch feature data based on DBSCAN clustering. Only
    takes place when there are more than 400 epochs for the subject.

    :param df_numeric:
    :param df_plot:
    :return:
    """
    # hard-coded cutoff for animals that already have a relatively low amount of epochs
    if df_numeric.shape[0] < 400:
        df_plot['cluster'] = np.ones(len(df_numeric), dtype=int)
        return df_plot

    dbscan = DBSCAN(eps=2.4, min_samples=35)
    outlier_preds = dbscan.fit_predict(df_numeric)

    df_plot['cluster'] = outlier_preds
    print("Number of outliers: ", len(outlier_preds[outlier_preds == -1]))

    return df_plot


def engineer_features(non_mov_epochs, wanted_chans):
    """
    Engineers desired features for the given subject. If there are two quality EMG channels,
    then the features derived from both EMG channels are averaged.

    :param non_mov_epochs: the subject's epochs
    :param wanted_chans: indexes used to retrieve right epoch data (one EEG channel, and one EMG channels)
    :return:
    """
    print(f'Engineering features..')

    sfreq = non_mov_epochs.info['sfreq']
    all_features = []  # list holding dict with all features per epoch

    # loop through the epochs in the subject's epochs
    for i, epoch in enumerate(non_mov_epochs):
        # features per epoch are stored in here
        features = {
            'subject_id': non_mov_epochs.metadata['animal_id'].iloc[0],
            'epoch_n': non_mov_epochs.metadata.iloc[i].name,
            'movement': non_mov_epochs.metadata["movement"].iloc[i],
            'frame_start_end': non_mov_epochs.metadata["epochs_start_end_frames"].iloc[i]
        }

        # for all wanted_chans, calculate the desired features
        for chan_type, chan_index in wanted_chans.items():
            if chan_type == 'EEG':
                # we only wish to calc PSD features using one chan, so get first index and then the data
                eeg_chan_data = epoch[chan_index, :]
                # get eeg psd features with right data from epoch
                features.update(calculate_eeg_psd_features(eeg_chan_data, sfreq))

            # if the looped channel type we need to calc features for is EMG, then we end up here
            if chan_type == 'EMG':
                # if there's only one EMG channel, save its features
                channel_data = epoch[chan_index, :]
                features.update(calculate_emg_psd_features(channel_data, sfreq))

        # store this epoch's features in the list
        all_features.append(features)

    print(f'Done engineering features.')

    return pd.DataFrame(all_features)


def get_wanted_channels(subject_epochs, wanted_eeg_chan, wanted_emg_chan):
    """
    Generates a structure that is used by the feature engineering function. More channel
    names could be added somehow by user if desired.

    :param subject_epochs:
    :param wanted_eeg_chan:
    :param wanted_emg_chan:
    :return:
    """
    recorded_chans = subject_epochs.info['ch_names']
    wanted_chan_indexes = {
        'EEG': [index for index, value in enumerate(recorded_chans) if value == wanted_eeg_chan][0],
        'EMG': [index for index, value in enumerate(recorded_chans) if value == wanted_emg_chan][0]
    }
    return wanted_chan_indexes


def classify_and_save_epochs(subject_epochs, subject_id):
    """
    Generates the desired features for this subject. Each epoch for this subject is assigned
     to one of the three desired clusters. Hypothetically, the non-movement epochs are clustered
     into sleep, active and 'resting-state' groups.

    :param subject_epochs:
    :param subject_id:
    :return:
    """
    wanted_eeg_chan, wanted_emg_chan = 'OFC_L', quality_emg[int(subject_id)]
    wanted_chan_indexes = get_wanted_channels(subject_epochs, wanted_eeg_chan, wanted_emg_chan)

    # engineer features
    non_mov_epochs = subject_epochs[subject_epochs.metadata["movement"] == 0]
    print(f'Total epochs: {len(subject_epochs)}, of which {len(non_mov_epochs)} are non-movement.')
    features = engineer_features(non_mov_epochs, wanted_chan_indexes)

    # get the numerical features from the created features dataframe (i.e. get rid of the epoch # etc.)
    df_numeric = features.iloc[:, 4:]

    # scale the features
    scaler = MinMaxScaler().set_output(transform="pandas")
    scaled_features = scaler.fit_transform(df_numeric)

    # perform principal component analysis
    pca = PCA(n_components=2)
    comp = pca.fit_transform(scaled_features)

    # save reduced dimensions and remove outliers
    df_plot = pd.concat([features.reset_index(drop=True), pd.DataFrame(comp)], axis=1)
    df_plot = tag_outliers(df_numeric, df_plot)  # remove outliers using DBSCAN
    df_numeric_wo_outliers = df_numeric[~df_plot['cluster'].isin([-1])]  # df to be rescaled
    scaled_features = scaler.fit_transform(df_numeric_wo_outliers)  # rescale non-outlier epoch features

    # perform clustering
    kmeans = KMeans(random_state=40, n_clusters=3)
    kmeans.fit(scaled_features)
    print(f'There are 3 clusters with sizes: {np.unique(kmeans.labels_, return_counts=True)[1]}')

    # add cluster labels to df_plot
    outlier_indexes = df_plot.index.difference(df_numeric_wo_outliers.index)
    non_outlier_indexes = df_plot.index.difference(outlier_indexes)
    df_plot.loc[non_outlier_indexes, 'cluster'] = kmeans.labels_  # add the cluster labels to the non_outlier indexes

    # save the grid plot that visualizes the characteristics of each cluster for this subject
    save_radar_cluster_plot(scaled_features, df_plot[df_plot['cluster'] != -1], subject_id)

    # store the cluster column of df_plot as a numpy array in the metadata of the subject's epoch object and
    # save it to the filesystem such that it can be analyzed later
    non_mov_epochs.metadata["cluster"] = np.array(df_plot['cluster'])
    non_mov_epochs.save(
        os.path.join(paths['epochs_folder'], f"filtered_epochs_w_clusters_{subject_id}-epo.fif"),
        overwrite=True
    )


def main():
    # classify non-movement epochs per subject
    for epochs_filename in os.listdir(paths['epochs_folder']):
        if not epochs_filename.startswith('filtered_epochs_w_movement_') or not epochs_filename.endswith('epo.fif'):
            continue

        # load the epochs of this subject
        subject_id = epochs_filename.split('_')[-1].split('-')[0]

        # for now, skipping the subjects that are of bad quality or seem to need clustering using 4 clusters
        print(f"Working with subject {subject_id}.")

        subject_epochs = mne.read_epochs(os.path.join(paths['epochs_folder'], epochs_filename), preload=True)
        subject_epochs = subject_epochs[:-1]  # somehow the last epoch holds only zeros

        classify_and_save_epochs(subject_epochs, subject_id)

        print(f"Done with subject {subject_id}.\n")

    print("Done, bye.")


if __name__ == "__main__":
    main()
