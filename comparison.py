from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    )
import tensorflow as tf

from datetime import timedelta
from time import perf_counter

from os.path import join
from os import makedirs, listdir, environ

from natsort import natsorted

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

import metrics

from collections import Counter


def onehot_encoder(feature):
    """
    Encode categorical features as a one-hot numeric array
    """
    ohe = OneHotEncoder(sparse=False)
    feature_vector = ohe.fit_transform(feature)

    return feature_vector


def listdir_nohidden(path):
    """
    List files in directory ignoring hidden files (starting with a point)

    Args:
        path (str): Path to directory

    Yields:
        generator object
    """
    for file in natsorted(listdir(path)):
        if not file.startswith('.'):
            yield file


def load_data_as_dict(data_dir):
    """Loads Monte Carlo cross-validation (MCCV) data into a data dictionary

    Args:
        data_dir (str): Path to directory containing all MCCV folds

    Returns:
        dictionary: Data dictionary containing the fold path, fold number,
                    training dataset path (trainF), training label dataset
                    path (trainL), validation dataset path (validateF) and
                    validation label dataset path (validateL), test dataset
                    path (testF) and test label dataset path (testL).
    """
    data_dict = {'fold_path': [], 'fold': [], 'trainF': [], 'trainL': [], 'testF': [],
                 'testL': [], 'validateF': [], 'validateL': []}

    for fold in listdir_nohidden(data_dir):
        data_dict['fold'].append(fold)
        data_dict['fold_path'].append(join(data_dir, fold))

        for dataset in listdir_nohidden(join(data_dir, fold)):
            if dataset == "TrainF.csv":
                data_dict['trainF'].append(
                    join(data_dir, fold, dataset))
            if dataset == "TrainL.csv":
                data_dict['trainL'].append(
                    join(data_dir, fold, dataset))
            if dataset == "TestF.csv":
                data_dict['testF'].append(
                    join(data_dir, fold, dataset))
            if dataset == "TestL.csv":
                data_dict['testL'].append(
                    join(data_dir, fold, dataset))
            if dataset == "ValidateF.csv":
                data_dict['validateF'].append(
                    join(data_dir, fold, dataset))
            if dataset == "ValidateL.csv":
                data_dict['validateL'].append(
                    join(data_dir, fold, dataset))

    return data_dict


def make_model(input_shape, hidden_neuron_count):
    """Makes neural network model

    Args:
        input_shape (tuple): Input shape of to model
        hidden_neuron_count (int): Number of neurons in hidden layer

    Returns:
        Compiled model
    """
    input = Input(shape=input_shape, name="Input")

    dense1 = Dense(hidden_neuron_count,
                   activation="LeakyReLU",
                   kernel_regularizer="l2",
                   name="Hidden")(input)
    output = Dense(2, activation="softmax", name="Output")(dense1) 

    nn = Model(inputs=input, outputs=output)
    nn.compile(optimizer="adam", loss="binary_crossentropy")    

    return nn


def run_comparison(data_dir, experiment_id, output_dir, hidden_neuron_count):
    """
    Main function, runs the entire pipeline
    """
    # Start counter
    start_time = perf_counter()

    # Make output directories
    experiment_dir = join(output_dir, experiment_id)
    makedirs(experiment_dir, exist_ok=True)

    # Load Monte Carlo cross-validation (MCCV) data as dictionary
    data_dict = load_data_as_dict(data_dir)

    # Get number of classes
    tld_df = pd.read_csv(data_dict["trainL"][0], sep=";")
    num_classes = len(tld_df.nunique())
    print(f"Number of classes: {num_classes}")

    # Get number of samples
    num_samples = len(tld_df)
    print(f"Number of samples: {num_samples}\n")

    # Set up classifier
    num_patience = 300
    min_delta = 0.05
    num_epochs = 5000
    validation_split = 0.2
    callback = [EarlyStopping(monitor="val_loss", patience=num_patience,
                              min_delta=min_delta)]

    performance_foldwise = {"sns": [], "spc": [], "ppv": [], "npv": [], "acc": []}

    # Do MCCV
    for fold_index, _ in enumerate(data_dict["fold_path"]):
        print(f"Experiment ID: {experiment_id}")
        print(f"Fold: {fold_index+1}/{len(data_dict['fold'])}")

        # Read and assign data
        X_train = pd.read_csv(data_dict['trainF'][fold_index], delimiter=";",
                              index_col=0).values
        X_test = pd.read_csv(data_dict['testF'][fold_index], delimiter=";",
                             index_col=0).values
        y_train = pd.read_csv(data_dict['trainL'][fold_index], delimiter=";",
                              index_col=0).values
        y_test = pd.read_csv(data_dict['testL'][fold_index], delimiter=";",
                             index_col=0).values.flatten()

        # Read and assing validation data if present
        if data_dict['validateF']:            
            X_val = pd.read_csv(data_dict['validateF'][fold_index], delimiter=";",
                                index_col=0).values
            y_val = pd.read_csv(data_dict['validateL'][fold_index], delimiter=";",
                                index_col=0).values
     
        # Get number of positive and negative samples   
        sample_count = Counter(y_train.flatten())
        minority_class = min(sample_count, key=sample_count.get)
        majority_class = max(sample_count, key=sample_count.get)
        majority_count = sample_count[majority_class]

        beta = majority_count/num_samples
        class_weight = {minority_class: beta, majority_class: 1-beta}
        
        # Do one-hot encoding
        y_train = onehot_encoder(y_train)      

        # Make model
        nn = make_model(input_shape=X_train.shape[1:],
                        hidden_neuron_count=hidden_neuron_count)   

        # Fit
        if data_dict['validateF']:
            y_val = onehot_encoder(y_val)      
            history = nn.fit(x=X_train,
                             y=y_train,
                             batch_size=min(200, num_samples),
                             epochs=num_epochs,
                             callbacks=[callback],
                             shuffle=False,
                             validation_data=(X_val, y_val),
                             class_weight=class_weight,
                             verbose=0)
        else:
            history = nn.fit(x=X_train,
                             y=y_train,
                             batch_size=min(200, num_samples),
                             epochs=num_epochs,
                             callbacks=[callback],
                             shuffle=True,
                             validation_split=validation_split,
                             class_weight=class_weight,
                             verbose=0)

        # Predict on test data
        y_pred_raw = nn.predict(X_test)

        y_pred = []
        for pred in y_pred_raw:
            y_pred.append(pred.argmax())

        # Calculate metrics
        acc = metrics.accuracy(y_test, y_pred)
        sns = metrics.sensitivity(y_test, y_pred)
        spc = metrics.specificity(y_test, y_pred)
        ppv = metrics.positive_predictive_value(y_test, y_pred)
        npv = metrics.negative_predictive_value(y_test, y_pred)

        # Append performance to fold-wise and overall containers
        performance_foldwise["acc"].append(acc)
        performance_foldwise["sns"].append(sns)
        performance_foldwise["spc"].append(spc)
        performance_foldwise["ppv"].append(ppv)
        performance_foldwise["npv"].append(npv)

        # Print fold-wise results
        print(f"ACC: {acc:.4f}")
        print(f"SNS: {sns:.4f}")
        print(f"SPC: {spc:.4f}")
        print(f"PPV: {ppv:.4f}")
        print(f"NPV: {npv:.4f}")
        print()

    # Save performances
    performance_foldwise_df = pd.DataFrame(performance_foldwise)
    avg_values = pd.Series(performance_foldwise_df.mean(axis=0))
    performance_foldwise_df = performance_foldwise_df.append(avg_values, ignore_index=True)
    data_dict["fold"].append("Average")
    performance_foldwise_df.index = data_dict["fold"]
    performance_foldwise_df.to_csv(join(experiment_dir, f"performance_nn.csv"), sep=";")
    print(performance_foldwise_df.round(2))

    # Report run time
    end_time = perf_counter()
    run_time = end_time - start_time
    print("-" * 40)
    print(f"Run time in hh:mm:ss.us: {timedelta(seconds=run_time)}")


if __name__ == "__main__":
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)
    environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging

    output_dir = "results"
    makedirs(output_dir, exist_ok=True)

    data_dir = [
        "data/Hearth/Dataset",
        "data/Mammo/Dataset",
        "data/BreastMNIST/Dataset",
        "data/PneumoniaMNIST/Dataset"
        ]
    experiment_id = [
        "Hearth",
        "Mammo",
        "BreastMNIST",
        "PneumoniaMNIST"
        ]

    for path_to_dataset, exp_id in zip(data_dir, experiment_id):
        if exp_id == "BreastMNIST" or exp_id == "PneumoniaMNIST":
            hidden_neuron_count = 1000
        else:
            hidden_neuron_count = 500
        print(exp_id, hidden_neuron_count)
        run_comparison(path_to_dataset, exp_id, output_dir,
                       hidden_neuron_count=hidden_neuron_count)