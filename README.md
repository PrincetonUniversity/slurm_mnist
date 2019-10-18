# Installing and Running TensorFlow on the HPC Clusters

[TensorFlow](https://www.tensorflow.org) is a popular deep learning library for training artificial neural networks. The installation instructions depend on the cluster:

## Version 2.0

### TigerGPU or Adroit

TensorFlow 2.0 is not available on Anaconda Cloud yet. However, it is available via PyPI. Follow these installation directions:

```
module load anaconda3
conda create --name tf2-gpu python=3.7
conda activate tf2-gpu
pip install tensorflow-gpu
```

Test the installation by running a short job. First, download the necessary data:

```
python -c "import tensorflow as tf; tf.keras.datasets.mnist.load_data()"
```

The above command will download the file mnist.npz to the directory `~/.keras/datasets`. Below is our TensorFlow script (`mnist2_classify.py`) which trains a classifier on the MNIST data set:

```
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

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

One must include `module load cudnn` in the Slurm script or `libcudnn.so.7` will not be found. Here is a sample Slurm script (`mnist2.cmd`):

```
#!/bin/bash
#SBATCH --job-name=tf2-test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)

module load anaconda3 cudnn
conda activate tf2-gpu

srun python mnist2_classify.py
```

Submit the job with `sbatch mnist.cmd`.

## Version 1.x

### TigerGPU or Adroit

```
module load anaconda3
conda create --name tf-gpu tensorflow-gpu
conda activate tf-gpu
```

Be sure to include `conda activate tf-gpu` and `#SBATCH --gres=gpu:1` in your Slurm script.

### Traverse

```
module load anaconda3
conda create --name=tf-gpu --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda tensorflow-gpu
conda activate tf-gpu
# accept the license agreement if asked
```

Be sure to include `conda activate tf-gpu` and `#SBATCH --gres=gpu:1` in your Slurm script.

### Perseus, Della or TigerCPU

There are two popular CPU-only versions of TensorFlow. One is provided by Anaconda:

```
module load anaconda3
conda create --name tf-cpu tensorflow
conda activate tf-cpu
```

The second is provided by Intel:

```
module load anaconda3
conda create --name tf-cpu --channel intel tensorflow
conda activate tf-cpu
```

Be sure to include `conda activate tf-cpu` in your Slurm script.

## Example

This example is meant to be a repackage of one of the basic TensorFlow tutorials
for use on one of Princeton University's HPC clusters. It gives a basic recipe
for how to work around a few of the things that make adapting these a challenge.
And it will help to make sure you have a working GPU environment.

## First, read the actual tutorial
This adapts one exercise from the TensorFlow/Keras [basic classification tutorial](https://www.tensorflow.org/tutorials/keras/basic_classification).

## Clone the repo

Log in to a head node on one of the GPU clusters (Tiger or Adroit).
Then clone the repo to your home directory using:

```
git clone https://github.com/bwhicks/slurm_mnist.git
```

This will get you the file in a folder called `slurm_mnist`

## Make an appropriate conda environment

Getting GPU (and some nice MKL) support for Tensorflow is as easy as:

```
# adroit or tigergpu
module load anaconda3
conda create --name tf-gpu matplotlib tensorflow-gpu
```

Once this command completes, as long as you have the `anaconda3` module loaded (current session only,
you'll note that we load it in the Slurm script `mnist.cmd`),
you'll have access to `conda` and can use it to access the Python
virtual environment you just created.
Note that matplotlib is required for this tutorial but it is not needed in general and can be omitted.

Activate the conda environment:

```
conda activate tf-gpu
```

## Verify that you have a working GPU enabled Tensorflow

There are no GPUs on the head node of any of the clusters with GPU capacity,
so you'll need to get one via the scheduler. You can do that interactively with:

```
salloc --gres=gpu:1 --time=00:05:00
```

When your allocation is granted, you'll be moved to a compute node and can then
run:

```
module load anaconda3
conda activate tf-gpu
python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
```

This should print a GPU, indicating you have a working install.

## Running the example

This is now a two step process, because compute nodes do not have internet access
to download the Keras training set, so we'll be pickling it for later use.

From your home directory

```
module load anaconda3
conda activate tf-gpu
cd slurm_mnist
python mnist_download.py
```

This will download the data and pickle it to a binary file. This is appropriate
on the head node in terms of getting a dataset.

Now that you have the data (`mnist.pickle`), you can schedule the job.

Edit the line of `mnist.cmd` to remove the space in the `--mail-user` line
and add your Princeton NetID (or other email address)

Then from the `slurm_mnist` directory run:

```
sbatch mnist.cmd
```

This will request a GPU, 5 minutes of computing time, and queue the job. You
should receive a job number and can check if your job is running or queued
via `squeue -u <YourNetID>`.

You'll also receive start and finish emails.

Once the job runs, you'll have a `slurm-xxxxx.out` file in the `slurm_mnist` with
tensorflow's messages, and the one expected output, an example graph of images
with tensorflow's best guess (and whether it was right) for what article of
clothing they might be. Download it via `scp` (or connect using `ssh -Y` with an XWindows server and use `eog`) and take a look!

## Examining GPU utilization

To see how effectively your job is using the GPU, immediately after submiting the job run the following command:

```
squeue -u <your-username>
```

The rightmost column labeled "NODELIST(REASON)" gives the name of the node where your job is running. SSH to this node:

```
ssh tiger-iXXgYY
```

Once on the compute node run `watch -n 1 gpustat`. This will show you a percentage value indicating how effectively your code is using the GPU. The memory allocated to the GPU is also available. You will see that only about 16% of the GPU cores are utilized. For a simple tutorial like this, performance is not a concern. However, when you begin running your research codes you should repeat this analysis to ensure that any GPUs you use are being nearly fully utilized.

Type `Ctrl+C` to exit the `watch` screen. Type `exit` to return to the head node.

While there is a CPU-only version of Tensorflow, given that we have two GPU clusters (Tiger and Adroit) and the mathematics of deep learning are ideally suited to GPUs, it is advised to work with the GPU version. However, if you wish to work with the CPU-only version simply remove "-gpu" from the package name when creating the conda environment:

```
conda create --name tf-cpu matplotlib tensorflow
```

If you encounter any difficulties while installing TensorFlow on one of our HPC clusters then please send an email to <a href="mailto:cses@princeton.edu">cses@princeton.edu</a> or attend a <a href="https://researchcomputing.princeton.edu/education/help-sessions">help session</a>.
