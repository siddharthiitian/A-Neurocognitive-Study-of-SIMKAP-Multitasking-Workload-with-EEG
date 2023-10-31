import os
import h5py
import numpy as np
import pandas as pd
import scipy
from numpy import mean
from matplotlib import pyplot as plt
# !pip install antropy
# import antropy as an
from scipy.stats import skew, kurtosis
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# from sklearn.ensemble import ExtraTreesClassifier
# from keras import layers, models, regularizers
# from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
# from sklearn.metrics import roc_curve,auc, precision_score,recall_score,f1_score
Features_df = pd.read_csv('/content/extractedFeatures.csv')
Features_df.drop(Features_df.columns[0], axis = 1, inplace = True)      # In the csv file first column has number values from 0-2280 and carries redundadnt info.

#3 Data preparation
##3.1 Shuffling
Data = Features_df.sample(frac = 1)
features = Data[[x for x in Data.columns if x not in ["Label"]]]   # Data for training
Labels = Data['Label']                                            # Labels for training
Labels = Labels.astype('category')

splitRatio = 0.3
train, test = train_test_split(Data ,test_size=splitRatio,
                               random_state = 123, shuffle = True)  # Spilt to training and testing data

train_X = train[[x for x in train.columns if x not in ["Label"]]]   # Data for training
train_Y = train['Label']                                            # Labels for training

###4.5.2 Testing Data
test_X = test[[x for x in test.columns if x not in ["Label"]]]     # Data fo testing
test_Y = test["Label"]                                              # Labels for training

###4.5.3 Validation Data
x_val = train_X[:200]                                                # 50 Sample for Validation
partial_x_train = train_X[200:]
partial_x_train = partial_x_train.values

y_val = train_Y[:200]
y_val = to_categorical(y_val)
partial_y_train = train_Y[200:]
partial_y_train = partial_y_train.values
partial_y_train = to_categorical(partial_y_train)

print("Data is prepeared")

print("Start Building Classifer")



# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

# Hide the Configuration and Warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# Model = 'Long_Short_Term_Memory'
import random
import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Model as models
# from . import models
# from models.DatasetAPI.DataLoader import DatasetLoader
from models.Network.LSTM import LSTM
from models.Loss_Function.Loss import loss
from models.Evaluation_Metrics.Metrics import evaluation

# Model Hyper-parameters
n_input   = 64   # The input size of signals at each time
max_time  = 64   # The unfolded time slices of the LSTM Model
lstm_size = 256  # The number of RNNs inside the LSTM Model

n_class   = 4     # The number of classification classes
n_hidden  = 64    # The number of hidden units in the first fully-connected layer
num_epoch = 300   # The number of Epochs that the Model run
keep_rate = 0.75  # Keep rate of the Dropout

lr = tf.constant(1e-4, dtype=tf.float32)  # Learning rate
lr_decay_epoch = 50    # Every (50) epochs, the learning rate decays
lr_decay       = 0.50  # Learning rate Decay by (50%)

batch_size = 1024
n_batch = train_data.shape[0] // batch_size

# Initialize Model Parameters (Network Weights and Biases)
# This Model only uses Two fully-connected layers, and u sure can add extra layers DIY
weights_1 = tf.Variable(tf.truncated_normal([lstm_size, n_hidden], stddev=0.01))
biases_1  = tf.Variable(tf.constant(0.01, shape=[n_hidden]))
weights_2 = tf.Variable(tf.truncated_normal([n_hidden, n_class], stddev=0.01))
biases_2  = tf.Variable(tf.constant(0.01, shape=[n_class]))

# Define Placeholders
x = tf.placeholder(tf.float32, [None, 64 * 64])
y = tf.placeholder(tf.float32, [None, 4])
keep_prob = tf.placeholder(tf.float32)

# Load Model Network
prediction, features = LSTM(Input=x,
                            max_time=max_time,
                            n_input=n_input,
                            lstm_size=lstm_size,
                            keep_prob=keep_prob,
                            weights_1=weights_1,
                            biases_1=biases_1,
                            weights_2=weights_2,
                            biases_2=biases_2)

# Load Loss Function
loss = loss(y=y, prediction=prediction, l2_norm=True)

# Load Optimizer
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# Load Evaluation Metrics
Global_Average_Accuracy = evaluation(y=y, prediction=prediction)

# Merge all the summaries
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(SAVE + '/train_Writer', sess.graph)
test_writer = tf.summary.FileWriter(SAVE + '/test_Writer')

# Initialize all the variables
sess.run(tf.global_variables_initializer())
for epoch in range(num_epoch + 1):
    # U can use learning rate decay or not
    # Here, we set a minimum learning rate
    # If u don't want this, u definitely can modify the following lines
    learning_rate = sess.run(lr)
    if epoch % lr_decay_epoch == 0 and epoch != 0:
        if learning_rate <= 1e-6:
            lr = lr * 1.0
            sess.run(lr)
        else:
            lr = lr * lr_decay
            sess.run(lr)

    # Randomly shuffle the training dataset and train the Model
    for batch_index in range(n_batch):
        random_batch = random.sample(range(train_X.shape[0]), batch_size)
        batch_xs = train_X[random_batch]
        batch_ys = train_Y[random_batch]
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: keep_rate})

    # Show Accuracy and Loss on Training and Test Set
    # Here, for training set, we only show the result of first 100 samples
    # If u want to show the result on the entire training set, please modify it.
    train_accuracy, train_loss = sess.run([Global_Average_Accuracy, loss], feed_dict={x: train_data[0:100], y: train_labels[0:100], keep_prob: 1.0})
    Test_summary, test_accuracy, test_loss = sess.run([merged, Global_Average_Accuracy, loss], feed_dict={x: test_data, y: test_labels, keep_prob: 1.0})
    test_writer.add_summary(Test_summary, epoch)

    # Show the Model Capability
    print("Iter " + str(epoch) + ", Testing Accuracy: " + str(test_accuracy) + ", Training Accuracy: " + str(train_accuracy))
    print("Iter " + str(epoch) + ", Testing Loss: " + str(test_loss) + ", Training Loss: " + str(train_loss))
    print("Learning rate is ", learning_rate)
    print('\n')

    # Save the prediction and labels for testing set
    # The "labels_for_test.csv" is the same as the "test_label.csv"
    # We will use the files to draw ROC CCurve and AUC
    if epoch == num_epoch:
        output_prediction = sess.run(prediction, feed_dict={x: test_X, y: test_X, keep_prob: 1.0})
        np.savetxt(SAVE + "prediction_for_test.csv", output_prediction, delimiter=",")
        np.savetxt(SAVE + "labels_for_test.csv", test_X, delimiter=",")

    # if you want to extract and save the features from fully-connected layer, use all the dataset and uncomment this.
    # All data is the total data = training data + testing data
    # We use the features from the overall dataset
    # ML models might be used to classify the features further
    # if epoch == num_epoch:
    #     Features = sess.run(features, feed_dict={x: all_data, y: all_labels, keep_prob: 1.0})
    #     np.savetxt(SAVE + "Features.csv", features, delimiter=",")

train_writer.close()
test_writer.close()
sess.close()