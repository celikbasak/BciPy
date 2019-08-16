import pickle
import logging

from bcipy.helpers.load import (
    read_data_csv,
    load_experimental_data,
    load_json_parameters)
from bcipy.signal.evaluate import Evaluator
from bcipy.signal.process.filter import bandpass, notch, downsample
from bcipy.signal.model.mach_learning.train_model import train_pca_rda_kde_model
from bcipy.helpers.task import trial_reshaper
from bcipy.helpers.vizualization import generate_offline_analysis_screen
from bcipy.helpers.triggers import trigger_decoder
from bcipy.helpers.acquisition import analysis_channels,\
    analysis_channel_names_by_pos
from bcipy.helpers.stimuli import play_sound

import numpy as np

# Configure logging correctly
log = logging.getLogger(__name__)


def offline_analysis(data_folder: str = None,
                     parameters: dict = {}, alert_finished: bool = True):
    """ Gets calibration data and trains the model in an offline fashion.
        pickle dumps the model into a .pkl folder
        Args:
            data_folder(str): folder of the data
                save all information and load all from this folder
            parameter(dict): parameters for running offline analysis
            alert_finished(bool): whether or not to alert the user offline analysis complete

        How it Works:
        - reads data and information from a .csv calibration file
        - reads trigger information from a .txt trigger file
        - filters data
        - reshapes and labels the data for the training procedure
        - fits the model to the data
            - uses cross validation to select parameters
            - based on the parameters, trains system using all the data
        - pickle dumps model into .pkl file
        - generates and saves offline analysis screen
        - [optional] alert the user finished processing
    """

    if not data_folder:
        data_folder = load_experimental_data()

    mode = 'calibration'
    trial_length = parameters.get('collection_window_after_trial_length')

    raw_dat, _, channels, type_amp, fs = read_data_csv(
        data_folder + '/' + parameters.get('raw_data_name', 'raw_data.csv'))

    log.info(f'Channels read from csv: {channels}')
    log.info(f'Device type: {type_amp}')

    downsample_rate = parameters.get('down_sampling_rate', 2)

    # Remove 60hz noise with a notch filter
    notch_filter_data = notch.notch_filter(raw_dat, fs, frequency_to_remove=60)

    # bandpass filter from 2-45hz
    filtered_data = bandpass.butter_bandpass_filter(
        notch_filter_data, 2, 45, fs, order=2)

    # downsample
    data = downsample.downsample(filtered_data, factor=downsample_rate)

    # Process triggers.txt
    triggers_file = parameters.get('trigger_file_name', 'triggers.txt')
    _, t_t_i, t_i, offset = trigger_decoder(
        mode=mode,
        trigger_path=f'{data_folder}/{triggers_file}')

    static_offset = parameters.get('static_trigger_offset', 0)

    offset = offset + static_offset

    # Channel map can be checked from raw_data.csv file.
    # read_data_csv already removes the timespamp column.
    channel_map = analysis_channels(channels, type_amp)

    # trial data is 
    x, y, _, _ = trial_reshaper(t_t_i, t_i, data,
                                mode=mode, fs=fs, k=downsample_rate,
                                offset=offset,
                                channel_map=channel_map,
                                trial_length=trial_length)

    print(np.amax(x))
    print(np.amin(x))

    x_trial, y_trial, tr_trial = _remove_bad_data_by_trial(x, y, parameters, False)
    trials_per_sequence = parameters['stim_length']
    x_sequence, y_sequence, tr_sequence = _remove_bad_data_by_sequence(x, y, parameters, trials_per_sequence, False)

    k_folds = parameters.get('k_folds', 10)

    if tr_trial != 0:
        model, auc_trial = train_pca_rda_kde_model(x_trial, y_trial, k_folds=k_folds)
        print("AUC Trial: " + str(auc_trial))
    if tr_sequence != 0:
        model, auc_sequence = train_pca_rda_kde_model(x_sequence, y_sequence, k_folds=k_folds)
        print("AUC Sequence: " + str(auc_sequence))


    # log.info('Saving offline analysis plots!')

    # # After obtaining the model get the transformed data for plotting purposes
    # model.transform(x)
    # generate_offline_analysis_screen(
    #     x, y, model=model, folder=data_folder,
    #     down_sample_rate=downsample_rate,
    #     fs=fs, save_figure=True, show_figure=False,
    #     channel_names=analysis_channel_names_by_pos(channels, channel_map))

    # log.info('Saving the model!')
    # with open(data_folder + f'/model_{auc}.pkl', 'wb') as output:
    #     pickle.dump(model, output)

    # if alert_finished:
    offline_analysis_tone = parameters.get('offline_analysis_tone')
    play_sound(offline_analysis_tone)

    # return model

def _remove_bad_data_by_trial(trial_data, trial_labels, parameters,estimate):

    """ Removes bad data in a trial-by-trial fashion. Offline artifact rejection.
        Args:
        trial_data: a multidimensional array of Channels x Trials (chopped into 500ms chunks) x Voltages 
        trial_labels: an ndarray of 0s (non-targets) and 1s (target), each representing a trial
        parameter(dict): parameters to pull information for enabled rules and threshold values

     """
    # get enabled rules
    high_voltage_enabled = parameters['high_voltage_threshold']
    low_voltage_enabled = parameters['low_voltage_threshold']
    rejection_threshold = parameters['rejection_threshold']

    # invoke evaluator / rules
    evaluator = Evaluator(parameters, high_voltage_enabled, low_voltage_enabled)

    # iterate over trial data, evaluate the trials, remove if needed and modify the trial labels to reflect
    channel_number = trial_data.shape[0]
    trial_number = trial_data[0].shape[0]
    num_trials = trial_data[0].shape[0]

    trial = 0
    rejected_trials = 0
    rejection_suggestions = 0
    bad_channel_threshold = 1

    while trial < trial_number:
        # go channel-wise through trials
        for ch in range(channel_number):
            data = trial_data[ch][trial]
            # evaluate voltage samples from this trial
            response = evaluator.evaluate(data) 
            if not response: # if False
                rejection_suggestions += 1 
                if rejection_suggestions >= bad_channel_threshold:
                    # if the evaluator rejects the data and we've reached
                    # the threshold, then delete the trial from each channel,
                    # adjust trial labels to follow suit, then exit the loop
                    trial_data = np.delete(trial_data, trial, axis=1)
                    trial_labels = np.delete(trial_labels, trial)
                    trial_number -= 1
                    rejected_trials += 1
                    break
        rejection_suggestions = 0 
        trial += 1

    percent_rejected = (rejected_trials / num_trials) * 100

    print('Number Rejected Trial-based: ' + str(rejected_trials))

    if percent_rejected > rejection_threshold:

        raise Exception(f'Percentage of data removed too high [{percent_rejected}]')

    else:
        return trial_data, trial_labels, percent_rejected

def _remove_bad_data_by_sequence(trial_data, trial_labels, parameters, trials_per_sequence,estimate):
    """Remove Bad Data By Sequence.
​
    Removes bad data in a sequence-by-sequence fashion. Offline artifact rejection.
        Args:
        trial_data: a multidimensional array of Channels x Trials (chopped into 500ms chunks) x Voltages
        trial_labels: an ndarray of 0s (non-targets) and 1s (target), each representing a trial
        parameter(dict): parameters to pull information for enabled rules and threshold values

    """

    # get enabled rules
    high_voltage_enabled = parameters['high_voltage_threshold']
    low_voltage_enabled = parameters['low_voltage_threshold']
    rejection_threshold = parameters['rejection_threshold']

    # invoke evaluator / rules
    evaluator = Evaluator(parameters, high_voltage_enabled, low_voltage_enabled)

    # iterate over trial data, evaluate the sequences, remove if needed and modify the trial labels to reflect
    channel_number = trial_data.shape[0]

    # looping criteria changes by number of trials in a sequence rather than 1
    # which is used in _remove_bad_data_by_trial
    trial_number = trial_data[0].shape[0]
    num_trials = trial_data[0].shape[0]

    trial = 0
    rejected_trials = 0

    while trial < trial_number:
        # go channel-wise through trials
        for ch in range(channel_number):
            sequence_end = trial + trials_per_sequence
            data = trial_data[ch][trial:sequence_end]
            # evaluate voltage samples from this sequence
            # by iterating through trials within the sequence
            response = evaluator.evaluate(data)
            if not response:
                # if the evaluator rejects the data and we've reached
                # the threshold, then delete the entire sequence from each channel,
                # adjust trial labels to follow suit, then exit the loop
                trial_data = np.delete(trial_data, np.s_[trial:sequence_end], axis=1)
                trial_labels = np.delete(trial_labels, np.s_[trial:sequence_end])
                trial_number -= trials_per_sequence
                rejected_trials += trials_per_sequence
                break

        rejection_suggestions = 0
        trial += trials_per_sequence

    percent_rejected = (rejected_trials / num_trials ) * 100

    print('Number Rejected Sequence-based: ' + str(rejected_trials))

    if percent_rejected > rejection_threshold:

        raise Exception(f'Percentage of data removed too high [{percent_rejected}]')

    else:
        return trial_data, trial_labels, percent_rejected
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', default=None)
    parser.add_argument('-p', '--parameters_file',
                        default='bcipy/parameters/parameters.json')
    args = parser.parse_args()

    print(f'Loading params from {args.parameters_file}')
    parameters = load_json_parameters(args.parameters_file,
                                      value_cast=True)
    offline_analysis(args.data_folder, parameters)
