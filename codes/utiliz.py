import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import timeseries_dataset_from_array



def get_all_csv_name(path):
    filename_list = []
    for folderName, subfolders, filenames in os.walk(path):
        for file_name in filenames:
            if '.csv' in file_name:
                filename_list.append(file_name)
    return filename_list


def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle=True,
    multi_horizon=True,
):
    """Creates tensorflow dataset from numpy array.

    """

    inputs = timeseries_dataset_from_array(
        #np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        data_array[:-forecast_horizon],
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
        
    )

    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(
        data_array[target_offset:][:,:,0], 
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
       
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()



def enconding_process(data_array: np.ndarray, transf: np.ndarray,train_size: float, val_size: float):
    #hour,day,week
    time_steps_hours = len(data_array[:,0])
    num_nodes = len(data_array[0,:])
    ci_encode = np.stack((data_array[24*7:-24], data_array[24*6:-24*2], data_array[:-24*8],
                             np.zeros((time_steps_hours-24*8,28)),np.zeros((time_steps_hours-24*8,28)) ) ,axis=2)
    transf_m = np.mean(transf,0)           
    for d in range(num_nodes):
        for s in range(num_nodes):
            ci_encode[:,d,3:5] = transf[24*7:-24,transf_m[:,d].argsort()[-2:],d]               
    num_time_steps = ci_encode.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    train_array = ci_encode[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (ci_encode[num_train : (num_train + num_val)] - mean) / std
    test_array = (ci_encode[(num_train + num_val) :] - mean) / std
    train_array[np.isnan(train_array)] = 0
    val_array[np.isnan(val_array)] = 0
    test_array[np.isnan(test_array)] = 0
    train_array[np.isinf(train_array)] = 0
    val_array[np.isinf(val_array)] = 0
    test_array[np.isinf(test_array)] = 0


    return train_array, val_array, test_array,mean,std




