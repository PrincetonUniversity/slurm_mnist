import pickle
from tensorflow import keras

# Compute nodes don't have internet, so download in advance and pickle the
# image training data

# adapted from https://www.tensorflow.org/tutorials/keras/basic_classification
# see notes there for how to actually work things

fashion_mnist = keras.datasets.fashion_mnist
print('Downloading mnist data to mnist.pickle...')
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()

with open('mnist.pickle', 'wb') as fp:
     pickle.dump((train_images, train_labels, test_images, test_labels), fp)


#@title MIT License
#
# Copyright (c) 2017 François Chollet
# Adaptation for Slurm use (c) 2018 Benjamin Hicks
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
