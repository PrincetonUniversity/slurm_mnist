# TensorFlow Performance (CPU-only)

This page provides examples of how to use TensorFlow on CPUs. TensorFlow has many CPU kernels that are multithreaded (e.g., matrix multiplication). This means they can take advantage of our multi-core CPUs. There is very limited support for using more than 1 node to train a TensorFlow model. All examples here only work on 1 node but can use all the CPU-cores on the node. In general, performance is better when GPUs are used so if you have access to GPU nodes then skip this page and start there.

## Installation

```
$ ssh adroit  # or another cluster
$ module load anaconda3
$ conda create --name tf2-cpu tensorflow=2.0 <package-2> <package-3> ... <package-N>
```

If you go over quota then run the `checkquota` command and follow the link at the bottom of the output to request more space. You can also use `--prefix` as shown <a href="https://github.com/PrincetonUniversity/gpu_programming_intro/tree/master/03_your_first_gpu_job">here</a>.

Here are selected packages that are installed:

```
...
blas               pkgs/main/linux-64::blas-1.0-mkl
...
intel-openmp       pkgs/main/linux-64::intel-openmp-2019.4-243
...
mkl                pkgs/main/linux-64::mkl-2019.4-243
mkl-service        pkgs/main/linux-64::mkl-service-2.3.0-py37he904b0f_0
mkl_fft            pkgs/main/linux-64::mkl_fft-1.0.15-py37ha843d7b_0
mkl_random         pkgs/main/linux-64::mkl_random-1.1.0-py37hd6b4f25_0
...
numpy              pkgs/main/linux-64::numpy-1.17.4-py37hc1035e2_0
numpy-base         pkgs/main/linux-64::numpy-base-1.17.4-py37hde5b4d6_0
...
scipy              pkgs/main/linux-64::scipy-1.3.1-py37h7c811a0_0
...
```

We see that several packages based on the Intel Math kernel Library (MKL) are installed with TensorFlow.

## Parallelism

The starting point for taking advantage of parallelism in TensorFlow on CPUs is to increase the number of threads via `cpus-per-task`. It is important to set `OMP_NUM_THREADS` in the Slurm script. One can also set tuning parameters of the Intel MKL-DNN library. Naively setting `ntasks` to a value greater than 1 will lead to the same code being run `ntasks` times instead of the work being divided over `ntasks` tasks. The same is true for `nodes`. To learn more about the limited support offered for working with multiple nodes see [MultiWorkerMirroredStrategy](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras). Various tips regarding Intel OpenMP environment variables and threading parameters for TensorFlow are described [here](https://software.intel.com/en-us/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference). [This page](https://support.nesi.org.nz/hc/en-gb/articles/360000997675) is also useful.

## Matrix Multiplication Example

Below is the TensorFlow script:

```python
from time import perf_counter

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

#tf.debugging.set_log_device_placement(True)
#print(tf.config.experimental.list_physical_devices())
#tf.config.threading.set_inter_op_parallelism_threads(0)
#tf.config.threading.set_intra_op_parallelism_threads(0)

times = []
N = 10000
x = tf.random.normal((N, N), dtype=tf.dtypes.float64)
for _ in range(5):
  t0 = perf_counter()
  y = tf.linalg.matmul(x, x)
  elapsed_time = perf_counter() - t0
  times.append(elapsed_time)
print("Execution time: ", min(times))
print("TensorFlow version: ", tf.__version__)
print(times)
```

Below is the Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=tf2-matmul    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=<num>    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

# set the number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# time in milliseconds each thread should wait after completing the execution of
# a parallel region before sleeping
export KMP_BLOCKTIME=0

# bind threads to cores
export KMP_AFFINITY=granularity=fine,compact,0,0

module load anaconda3
conda activate tf2-cpu

srun python mm.py
```

Make sure you replace `<num>` with a number in the Slurm script above. The measured timings are below:

| cpus-per-task (or threads)| execution time (s) | speed-up ratio |  parallel efficiency |
|:--------------------------:|:--------:|:---------:|:-------------------:|
| 1                          |  20.1    |   1.0     |   100%              |
| 2                          |  13.7    |   1.5     |   73%               | 
| 4                          |  7.1     |   2.8     |   71%               |
| 8                          |  3.1     |   6.5     |   81%               |
| 16                         |  1.8     |   11.2    |   70%               |
| 32                         |  1.0     |   20.1    |   63%               |

The calculation was performed on Adroit using a 10k x 10k matrix in double precision.

When the following line is added:

```
print(tf.config.experimental.list_physical_devices())
```

The corresponding output is

```
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:XLA_CPU:0', device_type='XLA_CPU')]
```

The above is consistent with TensorFlow labeling all available cores under `CPU:0`. This is true even with having 2 physical CPUs per node. Is it possible to assign 16 cores to `CPU:0` and 16 to `CPU:1`?

When 8 threads were used and the following lines are enabled:

```
tf.config.threading.set_inter_op_parallelism_threads(m)
tf.config.threading.set_intra_op_parallelism_threads(n)
```

The execution time was found to be largely insensitive to the values of `m` and `n`.

When `tf.debugging.set_log_device_placement(True)` is added we find `Executing op _MklMatMul in device` indicating that an Intel MKL library was used.

[Read more](https://software.intel.com/en-us/cpp-compiler-developer-guide-and-reference-intel-compiler-extension-routines-to-openmp) and [more](https://software.intel.com/en-us/cpp-compiler-developer-guide-and-reference-thread-affinity-interface-linux-and-windows) on `KMP_BLOCKTIME` and `KMP_AFFINITY`.

## MNIST Example with the Sequential API

Obtain the data:

```
python -c "import tensorflow as tf; tf.keras.datasets.mnist.load_data()"
```

Below is the simple MNIST example from the TensorFlow website:

```python
from __future__ import absolute_import, division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)
```

Below is the Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=tf2-mnist     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=<num>    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node
#SBATCH --time=00:02:00          # total run time limit (HH:MM:SS)

# set the number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# allow threads to transition quickly (Intel MKL-DNN)
export KMP_BLOCKTIME=0

# bind threads to cores (Intel MKL-DNN)
export KMP_AFFINITY=granularity=fine,compact,0,0

module load anaconda3
conda activate tf2-cpu

srun python mnist2_classify.py
```

Here are the timings:

| cpus-per-task (or threads)| execution time (s) | speed-up ratio |  parallel efficiency |
|:--------------------------:|:--------:|:---------:|:-------------------:|
| 1                          |  31     |   1.00     |   100%              |
| 2                          |  42     |   0.73     |    36%               | 
| 4                          |  37     |   0.84     |    21%               |
| 8                          |  41     |   0.76     |    9%               |

The use of multiple threads in this case leads to increased execution times. This may be because the neural network is quite small and there is an overhead penalty for using multiple threads.

## MNIST Example with the Subclassing API

Obtain the data:

```
python -c "import tensorflow as tf; tf.keras.datasets.mnist.load_data()"
```

Below is the simple MNIST example from the TensorFlow website:

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()
```

Below is the Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=tf2-sub       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=<num>    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,compact,0,0

module load anaconda3
conda activate tf2-cpu

srun python sub.py
```


Here are the timings:

| cpus-per-task (or threads)| execution time (s) | speed-up ratio |  parallel efficiency |
|:--------------------------:|:--------:|:---------:|:-------------------:|
| 1                          |  216     |   1.0     |   100%              |
| 2                          |  121     |   1.8     |   89%               | 
| 4                          |  75     |   2.9     |   72%               |
| 8                          |  68     |   3.2     |   40%               |
| 16                         |  61     |   3.5     |   22%               |
| 32                         |  40     |   5.4     |    17%               |

Execution times were taken from `seff` as the "Job Wall-clock time". The data was generated on Adroit. If `KMP_BLOCKTIME`
and `KMP_AFFINITY` are not set then the job runs very slowly.

## CIFAR10 Example

Obtain the data:

```bash
$ module load anaconda3
$ conda activate tf2-cpu
$ python -c "import tensorflow as tf; tf.keras.datasets.cifar10.load_data()"
```

Below is the TensorFlow script taken from the [tutorials](https://www.tensorflow.org/tutorials/images/cnn):

```python
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
```

Below is the Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=tf2-cifar     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=<num>    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

# set the number of threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# allow threads to transition quickly (Intel MKL-DNN)
export KMP_BLOCKTIME=0

# bind threads to cores (Intel MKL-DNN)
export KMP_AFFINITY=granularity=fine,compact,0,0

module load anaconda3
conda activate tf2-cpu

srun python cifar.py
```

Note that `MKL_NUM_THREADS` could be used instead of `OMP_NUM_THREADS`. The parallelism is believed to be coming from routines in the Intel MKL-DNN library. If the number of threads is not set then the job runs very slowly requiring more than 10 minutes to complete.

| cpus-per-task (or threads)| execution time (s) | speed-up ratio |  parallel efficiency |
|:--------------------------:|:--------:|:---------:|:-------------------:|
| 1                          |  189     |   1.0     |   100%              |
| 2                          |  179     |   1.1     |   53%               | 
| 4                          |  142     |   1.3     |   33%               |
| 8                          |  108     |   1.8     |   22%               |
| 16                         |  118     |   1.6     |   10%               |
| 32                         |  119     |   1.6     |    5%               |

Execution times were taken from `seff` as the "Job Wall-clock time". The data was generated on Adroit.

Can anything be done with `tf.distribute`? Right now only one device (`CPU:0`) is recognized.
