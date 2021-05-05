from generate_features import *
from prepocessing import *
from experiments_classification import *
from ica import *
from gb_lda import *

sensors = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

columns = ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
       'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10', 'Alpha_TP9',
       'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10', 'Beta_TP9', 'Beta_AF7',
       'Beta_AF8', 'Beta_TP10', 'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8',
       'Gamma_TP10']


"""columns = ['Delta_TP9', 'Delta_AF8', 'Delta_TP10', 'Theta_TP9', 'Theta_AF8', 'Theta_TP10', 'Alpha_TP9',
           'Alpha_TP10', 'Gamma_AF7']"""

# more_columns = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

# window_start = 80000
# window_end = 80768
# comp_start = 80000
# comp_end = 80768

"""
Pre-processing
"""
# alignment('muse2_file/muse', 'muse2_file/note', 'muse2_file/alignment_v3', columns+sensors+more_columns)

# labeling('muse2_file/alignment_v3', 'muse2_file/labeled_v3')

# compute_ica('muse2_file/labeled_v2/', sensors, window_start, window_end, comp_start, comp_end)

path_to_file = 'muse2_file/labeled_v3/'

# rolling_windows(path_to_file)

root = 'feature'

name_experiment = 'feature_EEG'

folder_experiments = root + '/' + name_experiment

to_save = folder_experiments + '/feature_single/'

check_folder_or_create(to_save)


"""
(path_to_file, to_save, sensors, columns=None, len_sub_window=0, fs=256, last_window=False, window=False, pad=False, 
greater_freq=False, num_freq=10, drop=1):
"""

"""
Feature extraction
"""
# generate_feature_sets_by_one_file(path_to_file, to_save, sensors, columns, greater_freq=True, num_freq=10, drop=1, pad=True)

to_save = folder_experiments + '/feature_all/'

check_folder_or_create(to_save)

# generate_feature_set_by_all_files(path_to_file, to_save, sensors, columns, greater_freq=True, num_freq=10, drop=1, pad=True)

"""
Classification
"""
root_folder = 'results/'+name_experiment
check_folder_or_create(root_folder)

path_features = folder_experiments + '/feature_all/' + 'features.csv'
path_info = folder_experiments + '/feature_all/indicator/' + '_indicator.csv'
path_indices = 'index.npz'

path_packages = 'Weka'

# java_start(path_packages)
# experiment_A(root_folder, path_features)
# experiment_B(root_folder, path_features, path_info, path_indices)


path_features = folder_experiments + '/feature_single'
# experiment_C(root_folder, path_features)
# experiment_D(root_folder, path_features)
# java_stop()


# Experiment B using sklearn
new_path_features = folder_experiments + '/feature_single/'
new_path_info = folder_experiments + '/feature_all/new_indicator/new_indicator.csv'
new_path_save = root_folder + '/sklearn/'
check_folder_or_create(new_path_save)
sk_classification(new_path_features, new_path_info, new_path_save)
