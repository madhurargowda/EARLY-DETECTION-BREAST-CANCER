from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from model import InputForm
from flask import Flask, render_template, request

def compute(a, b, c, d, e, z, g, h, i):
    # Preparing the data:
    data_file_name = 'breast-cancer-wisconsin.data.txt'

    first_line = "id,clump_thickness,unif_cell_size,unif_cell_shape,marg_adhesion,single_epith_cell_size,bare_nuclei,bland_chrom,norm_nucleoli,mitoses,class"
    with open(data_file_name, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(first_line.rstrip('\r\n') + '\n' + content)

    df = pd.read_csv(data_file_name)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    df['class'].replace('2', 0, inplace=True)
    df['class'].replace('4', 1, inplace=True)

    df.to_csv("combined_data.csv", index=False)

    # Data sets
    CANCER_TRAINING = "cancer_training.csv"
    CANCER_TEST = "cancer_test.csv"

    # Load datasets.
    training_set = tf.keras.utils.get_file(CANCER_TRAINING, ... )  # Specify how to load your data
    test_set = tf.keras.utils.get_file(CANCER_TEST, ... )  # Specify how to load your data

    # Assuming the features are 9 as per your data
    feature_columns = [tf.feature_column.numeric_column("features", shape=(9,))]
        
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=2,
        model_dir="/tmp/iris_model"
    )

    # Fit model.
    # You need to use a proper input function here for TensorFlow 2.x
    classifier.train(input_fn=lambda: tf.data.Dataset.from_tensor_slices((training_set.data, training_set.target)).batch(32), steps=2000)

    new_sample = np.array([[a, b, c, d, e, z, g, h, i]], dtype=np.float32)

    # Use the predict function correctly
    predictions = list(classifier.predict(input_fn=lambda: tf.data.Dataset.from_tensor_slices(new_sample).batch(1)))

    return "malignant" if predictions[0] == 1 else "benign"

if __name__ == '__main__':
    # Initialize the parameters
    a = 1  # Replace with actual values
    b = 2  # Replace with actual values
    c = 3  # Replace with actual values
    d = 4  # Replace with actual values
    e = 5  # Replace with actual values
    z = 6  # Replace with actual values
    g = 7  # Replace with actual values
    h = 8  # Replace with actual values
    i = 9  # Replace with actual values

    print(compute(a, b, c, d, e, z, g, h, i))
