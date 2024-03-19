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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from mne.time_frequency import psd_array_multitaper

from settings import paths, freq_bands_eeg, freq_bands_emg, quality_emg

from helper_functions import save_figure

palette = sns.color_palette("husl", 3)


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


def save_cluster_plot(plot_df, subject_id):
    """
    Save PCA plot where datapoints are colored by K-means cluster.

    :param plot_df:
    :param subject_id:
    :return:
    """
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x=0, y=1, hue="cluster", palette=palette)
    plt.title(f"Clusters by PCA components, non-movement epochs of subject {subject_id}")
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")

    save_figure(os.path.join(paths['plots_folder'], f'ploss_thresh_500/non_mov_clustering/cluster_plots/{subject_id}'))


def save_radar_cluster_plot(df_numeric, df_plot, subject_id):
    """
    Save a grid that includes a radar plot for each cluster that describes feature importance.
    Fourth subplot is the cluster scatterplot.

    :param df_numeric:
    :param subject_id:
    :param df_plot:
    :return:
    """
    scaler_minmax = MinMaxScaler()
    features = pd.DataFrame(scaler_minmax.fit_transform(df_numeric))
    features.columns = df_numeric.columns  # copy the column names from the original dataframe
    features["cluster"] = df_plot["cluster"]  # add cluster labels
    clusters = np.unique(df_plot["cluster"])  # get amount of clusters

    # determine # of rows and cols for the subplot
    row_len = int(np.ceil(len(clusters) / 2))
    col_len = int(np.ceil(len(clusters) / row_len))
    # generate gridplot with radars
    fig = make_subplots(
        rows=row_len, cols=col_len,
        specs=[[{'type': 'polar'}, {'type': 'polar'}], [{'type': 'polar'}, {'type': 'scatter'}]],
        horizontal_spacing=0.15, vertical_spacing=0.15,
        shared_yaxes=True, shared_xaxes=True,
        subplot_titles=["Cluster 0", "Cluster 1", "Cluster 2", "Clusters by PCA components, non-movement epochs"]
    )

    # add a subplot to the figure for each cluster
    max_value, cluster_info = 0, {}
    for i, cluster in enumerate(clusters):
        cluster_features = features[features['cluster'] == cluster]
        cluster_features = cluster_features.iloc[:, :-1]  # remove cluster column
        average_features = cluster_features.mean(axis=0)  # average the features in this cluster

        # remove non-intuitive radar components and save the info
        average_features = average_features.drop(['gamma-delta ratio', 'EMG high-low freq ratio'])
        cluster_info[i] = average_features

        # keep track of the largest value so that all subplots' axis their limit can be set to this value
        max_value = np.max(average_features) if np.max(average_features) > max_value else max_value

        # add subplot to the figure
        row, col = (i // 2) + 1, (i % 2) + 1
        fig.add_trace(go.Scatterpolar(
            r=average_features,
            theta=list(average_features.index),
            fill='toself',
            name=f'Cluster {i}'
        ), row=row, col=col)

    palette = sns.color_palette("husl", 3).as_hex()

    # add the cluster plot
    fig.add_trace(go.Scatter(
        x=df_plot[0], y=df_plot[1], mode="markers",
        marker=dict(color=df_plot["cluster"], colorscale=palette),
        showlegend=False
    ), row=2, col=2)
    fig.update_xaxes(title_text="Principle Component 1", row=2, col=2)
    fig.update_yaxes(title_text="Principle Component 2", row=2, col=2)

    # complete figure
    fig.update_layout(
        height=360 * row_len, width=500 * col_len,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
        colorway=palette
    )
    fig.update_annotations(yshift=20)
    fig.update_polars(radialaxis=dict(range=[0, max_value + .05]))
    fig.write_image(os.path.join(paths['plots_folder'], f'ploss_thresh_500/non_mov_clustering/radar_cluster_plot/{subject_id}.png'))

    return cluster_info


def save_average_signal_grid_plot(df_plot, subject_id, subject_epochs, wanted_eeg_chan, wanted_emg_chan):
    unique_clusters = df_plot["cluster"].unique()
    fig, axes = plt.subplots(
        ncols=2, nrows=len(unique_clusters),
        figsize=(20, 3 * len(unique_clusters)),
        sharex=True, sharey=True
    )
    axes = axes.ravel()

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

    # show anomalies
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_plot, x=0, y=1, hue="clusters_db")
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


def perform_clustering(df_numeric):
    """
    Performs clustering and forces 3 clusters. In theory, active, sleep and resting states.

    :param df_numeric:
    :return:
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_numeric)
    kmeans = KMeans(random_state=40, n_clusters=3)
    kmeans.fit(scaled_features)
    return kmeans


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
    features = engineer_features(non_mov_epochs, subject_id, subject_epochs.info['sfreq'], wanted_chan_indexes)

    # get the numerical features from the created features dataframe (i.e. get rid of the epoch # etc.)
    df_numeric = features.iloc[:, 4:]
    comp = perform_pca(df_numeric)

    # save reduced dimensions and remove outliers
    df_plot = pd.concat([features.reset_index(drop=True), pd.DataFrame(comp)], axis=1)
    df_numeric, df_plot = remove_cluster_outliers(df_numeric, df_plot, subject_id)  # remove outliers using DBSCAN

    # perform clustering and add cluster labels to df_plot
    kmeans = perform_clustering(df_numeric)
    df_plot["cluster"] = kmeans.labels_

    print(f'There are 3 clusters with sizes: {np.unique(kmeans.labels_, return_counts=True)[1]}')

    # save cluster and radar plots for this subject
    save_cluster_plot(df_plot, subject_id)
    save_average_signal_grid_plot(df_plot, subject_id, subject_epochs, wanted_eeg_chan, wanted_emg_chan)
    cluster_info = save_radar_cluster_plot(df_numeric, df_plot, subject_id)
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
