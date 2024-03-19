import os
import sys
import mne
import random
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from scipy.signal import hilbert
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from mne.time_frequency import psd_array_multitaper

from settings import paths, freq_bands_eeg, freq_bands_emg, quality_emg

from helper_functions import save_figure


def calculate_emg_psd_features(signal, sfreq):
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
    eeg_psds = {}
    for band, (start, end) in freq_bands_eeg.items():
        psd, freq = psd_array_multitaper(signal, fmin=start, fmax=end, sfreq=sfreq, n_jobs=-1)

        eeg_psds[f"OFC_L {band}"] = np.mean(np.log(psd))

    eeg_psds['gamma-delta ratio'] = (eeg_psds['OFC_L $\\gamma$'] / eeg_psds['OFC_L $\\delta$'])
    return eeg_psds
#
# def calculate_emg_psd_features(signal, sfreq):
#     psd, freq = psd_array_multitaper(signal, fmin=0, fmax=100, sfreq=sfreq, n_jobs=-1, verbose=False)
#
#     psds = {}  # slice psd data of epoch based on the defined bands
#     for band, (start, end) in freq_bands_emg.items():
#         slice_start, slice_end = int(start / 100 * len(freq)), int(end / 100 * len(freq))
#         psd_slice = psd[slice_start:slice_end]
#
#         psds[f"EMG {band}"] = np.mean(np.log(psd_slice))
#
#     psds['EMG high-low freq ratio'] = ((psds['EMG band1'] + psds['EMG band2']) / (psds['EMG band10'] + psds['EMG band9']))
#
#     return psds
#
#
# def calculate_eeg_psd_features(signal, sfreq):
#     eeg_psds = {}
#     for band, (start, end) in freq_bands_eeg.items():
#         psd, freq = psd_array_multitaper(signal, fmin=start, fmax=end, sfreq=sfreq, n_jobs=-1)
#
#         eeg_psds[f"OFC_L {band}"] = np.mean(np.log(psd))
#
#     eeg_psds['gamma-delta ratio'] = (eeg_psds['OFC_L $\\delta$'] / eeg_psds['OFC_L $\\gamma$'])
#     return eeg_psds


def save_cluster_plot(plot_df, subject_id):
    """
    Save PCA plot where datapoints are colored by K-means cluster.

    :param plot_df:
    :param subject_id:
    :return:
    """
    fig = plt.figure()
    palette = sns.color_palette("husl", 3)
    sns.scatterplot(data=plot_df, x=0, y=1, hue="cluster", palette=palette)
    plt.title(f"Clusters by PCA components, non-movement epochs of subject {subject_id}")
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")

    save_figure(os.path.join(paths['plots_folder'], f'ploss_thresh_500/non_mov_clustering/cluster_plots/{subject_id}'))


def save_radar_plot(df_numeric, subject_id, kmeans):
    """
    Save a radar plot that describes individual feature importance per cluster.

    :param df_numeric:
    :param subject_id:
    :param kmeans:
    :return:
    """
    scaler_minmax = MinMaxScaler()
    scaled_features_minmax = scaler_minmax.fit_transform(df_numeric)

    features = pd.DataFrame(scaled_features_minmax)
    features.columns = df_numeric.columns
    features["cluster"] = kmeans.labels_  # add cluster labels
    clusters = np.unique(kmeans.labels_)
    row_len = int(np.ceil(len(clusters) / 2))
    col_len = int(np.ceil(len(clusters) / row_len))

    spec = [{'type': 'polar'} for i in range(col_len)]
    specs = [spec for i in range(row_len)]

    # generate gridplot with radars
    fig = make_subplots(
        rows=row_len, cols=col_len, specs=specs,
        horizontal_spacing=0.15, vertical_spacing=0.08,
        shared_yaxes=True, shared_xaxes=True
    )

    cluster_info = {}
    max_value = 0
    for i, cluster in enumerate(clusters):
        cluster_features = features[features['cluster'] == cluster]

        row, col = (i // 2) + 1, (i % 2) + 1

        # remove cluster column and average the features for this cluster
        cluster_features = cluster_features.iloc[:, :-1]
        average_features = cluster_features.mean(axis=0)
        # remove non-intuitive radar components
        average_features = average_features.drop(['gamma-delta ratio', 'EMG high-low freq ratio'])
        cluster_info[i] = average_features
        feature_names = average_features.index

        if np.max(average_features) > max_value:
            max_value = np.max(average_features)

        fig.add_trace(go.Scatterpolar(
            r=average_features,
            theta=list(feature_names),
            fill='toself',
            name=f'Cluster {i} (# of epochs: {cluster_features.shape[0]})'
        ), row=row, col=col)

    # complete figure
    fig.update_layout(
        height=350 * row_len, width=500 * col_len,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
        colorway=sns.color_palette("husl").as_hex()
    )
    fig.update_polars(radialaxis=dict(range=[0, max_value + .05]))
    fig.write_image(os.path.join(paths['plots_folder'], f'ploss_thresh_500/non_mov_clustering/radar_plots/{subject_id}.png'))

    return cluster_info


def save_average_signal_grid_plot(df_plot, subject_id, subject_epochs, wanted_eeg_chan, wanted_emg_chan):
    unique_clusters = df_plot["cluster"].unique()
    fig, axes = plt.subplots(
        ncols=2, nrows=len(unique_clusters),
        figsize=(20, 3 * len(unique_clusters)),
        sharex=True, sharey=True
    )
    axes = axes.ravel()
    palette = sns.color_palette("husl", 3)

    i = 0
    for cluster in np.sort(unique_clusters):
        # get the cluster data from the dataframe
        cluster_data = df_plot[df_plot["cluster"] == cluster]

        for chan in [wanted_eeg_chan, wanted_emg_chan]:
            # get epoch indexes
            epoch_indexes = cluster_data["epoch_n"].unique()
            # get correct epoch data
            epoch_data = subject_epochs[epoch_indexes].get_data(picks=chan)
            # average the epochs of this channel and cluster
            mean = np.mean(epoch_data, axis=0)[0]

            colors = {0: palette[1], 1: palette[2], 2: palette[0]}

            axes[i].plot(mean, color=colors[cluster])
            axes[i].set_title(f"Cluster {cluster}, channel: {chan}")
            i += 1

    plt.tight_layout()
    save_figure(os.path.join(paths['plots_folder'], f'ploss_thresh_500/non_mov_clustering/average_emg_signals/{subject_id}'))


def perform_pca(df_numeric):
    """
    Performs PCA and clustering using this subject's numerical dataframe

    :param df_numeric: the numerical features,
    :return:
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_numeric)
    pca = PCA(n_components=2)
    comp = pca.fit_transform(scaled_features)

    return comp


def remove_cluster_outliers(df_numeric, df_plot, subject_id):
    dbscan = DBSCAN(eps=2.5, min_samples=25)
    clusters = dbscan.fit_predict(df_numeric)
    df_plot['clusters_db'] = clusters
    palette = sns.color_palette("husl", len(np.unique(clusters)))

    # show anomalies
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_plot, x=0, y=1, hue="clusters_db", palette=palette)
    plt.title("Clusters by PCA components, non-movement epochs")
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")
    save_figure(os.path.join(paths['plots_folder'], f'ploss_thresh_500/non_mov_clustering/removed_outliers/{subject_id}'))

    # remove outliers and return
    df_numeric_wo_outliers, df_plot_wo_anomalies = (df_numeric[~df_plot['clusters_db'].isin([-1])],
                                                    df_plot[~df_plot['clusters_db'].isin([-1])])
    print(f"Removed {df_numeric.shape[0] - df_numeric_wo_outliers.shape[0]} epochs from the data")

    return df_numeric_wo_outliers, df_plot_wo_anomalies


def engineer_features(non_mov_epochs, subject_id, sfreq, wanted_chans):
    """
    Engineers desired features for the given subject. If there are two quality EMG channels,
    then the features derived from both EMG channels are averaged.

    :param non_mov_epochs: the subject's epochs
    :param subject_id: the id of the subject
    :param sfreq: sampling frequency of the EEG
    :param wanted_chans: indexes used to retrieve right epoch data (one EEG channel, and one EMG channels)
    :return:
    """
    print(f'Engineering features..')

    all_features = []  # list holding dict with all features per epoch

    # loop through the epochs in the subject's epochs
    for i, epoch in enumerate(non_mov_epochs):
        # features per epoch are stored in here
        features = {
            'subject_id': subject_id,
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


def report_classification(cluster_info):
    average_emgs = {}
    cluster_gammas = {}
    for cluster_n, info in cluster_info.items():
        average_emgs[cluster_n] = np.mean(info[5:])
        cluster_gammas[cluster_n] = info['OFC_L $\gamma$']

    highest_av_emg_cluster = max(average_emgs, key=average_emgs.get)
    highest_gamma_cluster = max(cluster_gammas, key=cluster_gammas.get)

    for cluster_n, info in cluster_info.items():

        if (cluster_n == highest_av_emg_cluster) & (cluster_n == highest_gamma_cluster):
            print(f'Cluster {highest_av_emg_cluster} is likely active')
        elif (info['OFC_L $\gamma$'] < info['OFC_L $\delta$']) & (
                (info['EMG band10'] + info['EMG band9']) < (info['EMG band1'] + info['EMG band2'])):
            print(f'Cluster {cluster_n} is likely sleep')
        else:
            print(f'Cluster {cluster_n} is likely resting (or something else)')


def classify_and_save_epochs(subject_epochs, subject_id):
    """
    Generates the desired features for this subject. Each epoch for this subject is assigned
     to one of the three desired clusters. Hypothetically, the non-movement epochs are clustered
     into sleep, active and 'resting-state' groups.

    :param subject_epochs:
    :param subject_id:
    :return:
    """

    # get the indexes of the channels from which the features are derived
    wanted_eeg_chan = 'OFC_L'
    wanted_emg_chan = quality_emg[int(subject_id)]
    recorded_chans = subject_epochs.info['ch_names']
    wanted_chan_indexes = {
        'EEG': [index for index, value in enumerate(recorded_chans) if value == wanted_eeg_chan][0],
        'EMG': [index for index, value in enumerate(recorded_chans) if value == wanted_emg_chan][0]
    }

    # engineer features
    non_mov_epochs = subject_epochs[subject_epochs.metadata["movement"] == False]
    print(f'Total epochs: {len(subject_epochs)}, of which {len(non_mov_epochs)} are non-movement.')
    features = engineer_features(non_mov_epochs, subject_id, subject_epochs.info['sfreq'], wanted_chan_indexes)

    # get the numerical features from the created features dataframe (i.e. get rid of the epoch # etc.)
    df_numeric = features.iloc[:, 4:]

    # perform PCA dimensionality reduction and add PCA components 1 and 2 to df
    comp = perform_pca(df_numeric)
    df_plot = pd.concat([features.reset_index(drop=True), pd.DataFrame(comp)], axis=1)
    df_numeric, df_plot = remove_cluster_outliers(df_numeric, df_plot, subject_id)  # remove outliers using DBSCAN

    # scale the features again and add cluster labels to df_plot
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_numeric)
    kmeans = KMeans(random_state=40, n_clusters=3)
    kmeans.fit(scaled_features)
    df_plot["cluster"] = kmeans.labels_
    print(f'There are 3 clusters with sizes: {np.unique(kmeans.labels_, return_counts=True)[1]}')

    # save cluster and radar plots for this subject
    save_cluster_plot(df_plot, subject_id)
    cluster_info = save_radar_plot(df_numeric, subject_id, kmeans)
    save_average_signal_grid_plot(df_plot, subject_id, subject_epochs, wanted_eeg_chan, wanted_emg_chan)
    report_classification(cluster_info)

    # todo add cluster labels to the epochs object and save


def main():
    # classify non-movement epochs per subject
    for epochs_filename in os.listdir(paths['epochs_folder']):
        if not epochs_filename.startswith('filtered_epochs_w_movement_') or not epochs_filename.endswith('epo.fif'):
            continue

        # load the epochs of this subject
        subject_id = epochs_filename.split('_')[-1].split('-')[0]

        print(f"Working with subject {subject_id}.")

        subject_epochs = mne.read_epochs(os.path.join(paths['epochs_folder'], epochs_filename), preload=True)
        subject_epochs = subject_epochs[:-1]  # somehow the last epoch holds only zeros

        classify_and_save_epochs(subject_epochs, subject_id)

        print(f"Done with subject {subject_id}.\n")

    print("Done, bye.")


if __name__ == "__main__":
    main()
