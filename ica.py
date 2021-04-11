import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA


def check_input_art():
    while True:
        try:
            val = [int(x) for x in input("Insert the artefacts you want to remove, from 0 to 3, multiple values separated by comma:\n").split(',')]
        except ValueError:
            print("Please enter a valid set of values")
            continue

        if any(x not in range(0, 4) for x in val):
            print("Please enter values between 0 and 3")
            continue
        else:
            break
    return val


# Plot 4 channels
def channels_plot(dataset, win_start, win_end):
    fig, axs = plt.subplots(2, 2, figsize=(15, 7), sharex=True, sharey=True)
    axs = axs.ravel()
    plt.margins(x=0.001)
    # plt.ylim(100, 1800)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    axs[0].plot(dataset['RAW_AF7'].iloc[win_start:win_end], label='AF7', color='maroon')
    axs[0].legend(loc="upper right", fontsize=12)
    axs[1].plot(dataset['RAW_AF8'].iloc[win_start:win_end], label='AF8', color='darkorange')
    axs[1].legend(loc="upper right", fontsize=12)
    axs[2].plot(dataset['RAW_TP9'].iloc[win_start:win_end], label='TP9', color='silver')
    axs[2].legend(loc="upper right", fontsize=12)
    axs[3].plot(dataset['RAW_TP10'].iloc[win_start:win_end], label='TP10', color='darkgreen')
    axs[3].legend(loc="upper right", fontsize=12)
    plt.xlabel('Time [samples]', fontsize=14, labelpad=15)
    plt.ylabel('Voltage [\u03BCV]', fontsize=14, labelpad=15)
    plt.show()


# Compute ICA, plot components and remove artefacts
def compute_ica(path_to_file, sensors, win_start, win_end, ica_start, ica_end):
    datasets = [pd.read_csv(path_to_file + str(el)) for el in os.listdir(path_to_file)]  # sets of file raw
    for index in range(len(datasets)):
        channels_plot(datasets[index], win_start, win_end)
        print("Dataset number: " + str(index))
        print("Computing ICA...")
        ica = FastICA(n_components=4, random_state=0, tol=0.05)
        comps = ica.fit_transform(datasets[index][sensors])

        # ICA components plot
        fig, axs = plt.subplots(2, 2, figsize=(18, 13), sharex=True, sharey=True)
        fig.subplots_adjust(hspace=.4, wspace=0)
        axs = axs.ravel()

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel('Time [samples]', fontsize=14, labelpad=15)
        plt.title('Dataset: ' + str(index) + ' - Components', fontsize=14)

        for i in range(4):
            axs[i].plot(comps[ica_start:ica_end, i], color='slategrey')
            axs[i].set_title(str(i))

        plt.show()

        # Removing artefacts
        confirm = False
        while not confirm:
            artefacts = check_input_art()
            comps[:, artefacts] = 0
            restored = ica.inverse_transform(comps)

            fig, axs = plt.subplots(2, 2, figsize=(15, 7), sharex=True, sharey=True)
            axs = axs.ravel()
            plt.margins(x=0.001)
            # plt.ylim(600, 1500)
            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            axs[0].plot(datasets[index]['RAW_AF7'].iloc[win_start:win_end], label='AF7_pre', color='rosybrown')
            axs[0].plot(np.arange(win_start, win_end), restored[win_start:win_end, 1], label='AF7_post', color='maroon')
            axs[0].legend(loc="upper right", fontsize=12)
            axs[1].plot(datasets[index]['RAW_AF8'].iloc[win_start:win_end], label='AF8_pre', color='silver')
            axs[1].plot(np.arange(win_start, win_end), restored[win_start:win_end, 2], label='AF8_post', color='dimgray')
            axs[1].legend(loc="upper right", fontsize=12)
            axs[2].plot(datasets[index]['RAW_TP9'].iloc[win_start:win_end], label='TP9_pre', color='bisque')
            axs[2].plot(np.arange(win_start, win_end), restored[win_start:win_end, 0], label='TP9_post', color='darkorange')
            axs[2].legend(loc="upper right", fontsize=12)
            axs[3].plot(datasets[index]['RAW_TP10'].iloc[win_start:win_end], label='TP10_pre', color='lightgreen')
            axs[3].plot(np.arange(win_start, win_end), restored[win_start:win_end, 3], label='TP10_post', color='darkgreen')
            axs[3].legend(loc="upper right", fontsize=12)
            plt.xlabel('Time [samples]', fontsize=14, labelpad=15)
            plt.ylabel('Voltage [\u03BCV]', fontsize=14, labelpad=15)
            plt.title('Dataset: ' + str(index), fontsize=12)
            plt.show()
            choice = input("Do you confirm your choice? y/n\n")
            if choice == 'y':
                confirm = True
            else:
                comps = ica.fit_transform(datasets[index][sensors])

        datasets[index]['RAW_TP9'] = restored[:, 0]
        datasets[index]['RAW_AF7'] = restored[:, 1]
        datasets[index]['RAW_AF8'] = restored[:, 2]
        datasets[index]['RAW_TP10'] = restored[:, 3]
        datasets[index].to_csv('muse2_file/ica/ica' + str(index) + '.csv')


def compute_single_ica(win_start, win_end, ica_start, ica_end):
    dataset = pd.read_csv('muse2_file/labeled/1raw_f.csv')
    columns = ['RAW_AF7', 'RAW_AF8', 'RAW_TP9', 'RAW_TP10']
    dataset = dataset[columns]
    ica = FastICA(n_components=4, random_state=0, tol=0.05)
    comps = ica.fit_transform(dataset)

    # ICA components plot
    fig, axs = plt.subplots(2, 2, figsize=(18, 13), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=.4, wspace=0)
    axs = axs.ravel()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Time [samples]', fontsize=14, labelpad=15)

    for i in range(4):
        axs[i].plot(comps[ica_start:ica_end, i], color='slategrey')
        axs[i].set_title(str(i))

    plt.show()

    # Removing artefacts
    artefacts = check_input_art()
    comps[:, artefacts] = 0
    restored = ica.inverse_transform(comps)

    fig, axs = plt.subplots(2, 2, figsize=(15, 7), sharex=True, sharey=True)
    axs = axs.ravel()
    plt.margins(x=0.001)
    plt.ylim(600, 1200)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    axs[0].plot(dataset['RAW_AF7'].iloc[win_start:win_end], label='AF7_pre', color='rosybrown')
    axs[0].plot(np.arange(win_start, win_end), restored[win_start:win_end, 0], label='AF7_post', color='maroon')
    axs[0].legend(loc="upper right", fontsize=12)
    axs[1].plot(dataset['RAW_AF8'].iloc[win_start:win_end], label='AF8_pre', color='silver')
    axs[1].plot(np.arange(win_start, win_end), restored[win_start:win_end, 1], label='AF8_post', color='dimgray')
    axs[1].legend(loc="upper right", fontsize=12)
    axs[2].plot(dataset['RAW_TP9'].iloc[win_start:win_end], label='TP9_pre', color='bisque')
    axs[2].plot(np.arange(win_start, win_end), restored[win_start:win_end, 2], label='TP9_post', color='darkorange')
    axs[2].legend(loc="upper right", fontsize=12)
    axs[3].plot(dataset['RAW_TP10'].iloc[win_start:win_end], label='TP10_pre', color='lightgreen')
    axs[3].plot(np.arange(win_start, win_end), restored[win_start:win_end, 3], label='TP10_post', color='darkgreen')
    axs[3].legend(loc="upper right", fontsize=12)
    plt.xlabel('Time [samples]', fontsize=14, labelpad=15)
    plt.ylabel('Voltage [\u03BCV]', fontsize=14, labelpad=15)
    plt.show()
