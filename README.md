# Installing and Running TensorFlow on the HPC Clusters

[TensorFlow](https://www.tensorflow.org) is a popular deep learning library for training artificial neural networks. The installation instructions depend on the version and cluster:

# Version 2.x

### TigerGPU or Adroit

TensorFlow 2.0 is available on Anaconda Cloud:

```
$ module load anaconda3
$ conda create --name tf2-gpu tensorflow-gpu <package-2> <package-3> ... <package-N>
$ conda activate tf2-gpu
```

### TigerCPU, Della or Perseus

To install the CPU-only version of TensorFlow 2.0 follow the directions above except replacement `tensorflow-gpu` with `tensorflow` and `tf2-gpu` with `tf2-cpu`. In the Slurm script below you must remove the gpu line (`#SBATCH --gres=gpu:1`) and be sure to use the environment name `tf2-cpu`. For tips on running on CPUs see [this page](https://github.com/jdh4/tensorflow_performance).

### Traverse

A beta release of TensorFlow 2.0 is available for IBM's Power architecture and NVIDIA GPUs on IBM's Conda channel, Watson Machine Learning Community Edition. Follow these installation directions:

```bash
$ module load anaconda3
$ conda create --name=tf2-gpu --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda tensorflow2-gpu
$ conda activate tf2-gpu
```

## Example for Version 2.x

Test the installation by running a short job. First, download the necessary data:

```bash
$ python -c "import tensorflow as tf; tf.keras.datasets.mnist.load_data()"
```

The above command will download `mnist.npz` into the directory `~/.keras/datasets`. To run the example follow these commands:

```
$ git clone https://github.com/PrincetonUniversity/slurm_mnist.git
$ cd slurm_mnist
$ sbatch job.slurm
```

Below is the TensorFlow script (`mnist2_classify.py`) which trains a classifier on the MNIST data set:

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

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
```

Here is the Slurm script (`job.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=tf2-test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3
conda activate tf2-gpu

srun python mnist2_classify.py
```

# Version 1.x

### TigerGPU or Adroit

```
$ module load anaconda3
$ conda create --name tf-gpu tensorflow-gpu=1.15 <package-2> <package-3> ... <package-N>
$ conda activate tf-gpu
```

Be sure to include `conda activate tf-gpu` and `#SBATCH --gres=gpu:1` in your Slurm script.

### Traverse

```
$ module load anaconda3
$ conda create --name=tf-gpu --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda tensorflow-gpu
$ conda activate tf-gpu
# accept the license agreement if asked
```

Be sure to include `conda activate tf-gpu` and `#SBATCH --gres=gpu:1` in your Slurm script.

### Perseus, Della or TigerCPU

There are two popular CPU-only versions of TensorFlow. One is provided by Anaconda:

```
$ module load anaconda3
$ conda create --name tf-cpu tensorflow=1.15 <package-2> <package-3> ... <package-N>
$ conda activate tf-cpu
```

The second is provided by Intel:

```
$ module load anaconda3
$ conda create --name tf-cpu --channel intel tensorflow=1.15 <package-2> <package-3> ... <package-N>
$ conda activate tf-cpu
```

Be sure to include `conda activate tf-cpu` in your Slurm script. For tips on running on CPUs see [this page](https://github.com/jdh4/tensorflow_performance).

## Example for Version 1.x

This example is meant to be a repackage of one of the basic TensorFlow tutorials
for use on one of Princeton University's HPC clusters. It gives a basic recipe
for how to work around a few of the things that make adapting these a challenge.
And it will help to make sure you have a working GPU environment.

### First, read the actual tutorial
This adapts one exercise from the TensorFlow/Keras [basic classification tutorial](https://www.tensorflow.org/tutorials/keras/basic_classification).

### Clone the repo

Log in to a head node on one of the GPU clusters (Tiger or Adroit).
Then clone the repo to your home directory using:

```
$ git clone https://github.com/PrincetonUniversity/slurm_mnist.git
```

This will get you the file in a folder called `slurm_mnist`

### Make an appropriate conda environment

Getting GPU (and some nice MKL) support for Tensorflow is as easy as:

```
# adroit or tigergpu
$ module load anaconda3
$ conda create --name tf-gpu matplotlib tensorflow-gpu=1.15
```

Once this command completes, as long as you have the `anaconda3` module loaded (current session only,
you'll note that we load it in the Slurm script `mnist.cmd`),
you'll have access to `conda` and can use it to access the Python
virtual environment you just created.
Note that matplotlib is required for this tutorial but it is not needed in general and can be omitted.

Activate the conda environment:

```
$ conda activate tf-gpu
```

### Verify that you have a working GPU enabled Tensorflow

There are no GPUs on the head node of any of the clusters with GPU capacity,
so you'll need to get one via the scheduler. You can do that interactively with:

```
$ salloc -N 1 -n 1 --time=00:05:00 --gres=gpu:1
```

When your allocation is granted, you'll be moved to a compute node and can then
run:

```
$ module load anaconda3
$ conda activate tf-gpu
$ python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
```

This should print a GPU, indicating you have a working install.

### Running the example

This is now a two step process, because compute nodes do not have internet access
to download the Keras training set, so we'll be pickling it for later use.

From your home directory

```
$ module load anaconda3
$ conda activate tf-gpu
$ cd slurm_mnist
$ python mnist_download.py
```

This will download the data and pickle it to a binary file. This is appropriate
on the head node in terms of getting a dataset.

Now that you have the data (`mnist.pickle`), you can schedule the job.

Edit the line of `mnist.cmd` to remove the space in the `--mail-user` line
and add your Princeton NetID (or other email address)

Then from the `slurm_mnist` directory run:

```
$ sbatch mnist.cmd
```

This will request a GPU, 5 minutes of computing time, and queue the job. You
should receive a job number and can check if your job is running or queued
via `squeue -u <YourNetID>`.

You'll also receive start and finish emails.

Once the job runs, you'll have a `slurm-xxxxx.out` file in the `slurm_mnist` with
tensorflow's messages, and the one expected output, an example graph of images
with tensorflow's best guess (and whether it was right) for what article of
clothing they might be. Download it via `scp` (or connect using `ssh -Y` with an XWindows server and use `eog`) and take a look!

### Examining GPU utilization

To see how effectively your job is using the GPU, immediately after submiting the job run the following command:

```
$ squeue -u <your-username>
```

The rightmost column labeled "NODELIST(REASON)" gives the name of the node where your job is running. SSH to this node:

```
$ ssh tiger-iXXgYY
```

Once on the compute node run `watch -n 1 nvidia-smi`. This will show you a percentage value indicating how effectively your code is using the GPU. The memory allocated to the GPU is also available. You will see that only about 16% of the GPU cores are utilized. For a simple tutorial like this, performance is not a concern. However, when you begin running your research codes you should repeat this analysis to ensure that any GPUs you use are being nearly fully utilized.

Type `Ctrl+C` to exit the `watch` screen. Type `exit` to return to the head node.

## Running on Mulitiple GPUs on a Single Node

Models that are built using keras can be made to run on multiple GPUs quite easily. This is done by copying the model and having each GPU process a different mini-batch. For more see [this example](https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_keras).

## Using PyCharm on TigerGPU

<a href="http://www.youtube.com/watch?feature=player_embedded&v=0XmZsfixAdw" target="_blank">
<img src="http://img.youtube.com/vi/0XmZsfixAdw/0.jpg" alt="PyCharm" width="320" height="180" border="0" /></a>

## Where to Store Your Files

You should run your jobs out of `/scratch/gpfs/<NetID>` on the HPC clusters. These filesystems are very fast and provide vast amounts of storage. **Do not run jobs out of `/tigress` or `/projects`. That is, you should never be writing the output of actively running jobs to those filesystems.** `/tigress` and `/projects` are slow and should only be used for backing up the files that you produce on `/scratch/gpfs`. Your `/home` directory on all clusters is small and it should only be used for storing source code and executables.

The commands below give you an idea of how to properly run a TensorFlow job:

```
$ ssh <NetID>@tigergpu.princeton.edu
$ cd /scratch/gpfs/<NetID>
$ mkdir myjob && cd myjob
# put TensorFlow script and Slurm script in myjob
$ sbatch job.slurm
```

If the run produces data that you want to backup then copy or move it to `/tigress`:

```
$ cp -r /scratch/gpfs/<NetID>/myjob /tigress/<NetID>
```

For large transfers consider using `rsync` instead of `cp`. Most users only do back-ups to `/tigress` every week or so. While `/scratch/gpfs` is not backed-up, files are never removed. However, important results should be transferred to `/tigress` or `/projects`.

The diagram below gives an overview of the filesystems:

![tigress](https://tigress-web.princeton.edu/~jdh4/hpc_princeton_filesystems.png)

## Getting Help

If you encounter any difficulties while installing TensorFlow on one of our HPC clusters then please send an email to <a href="mailto:cses@princeton.edu">cses@princeton.edu</a> or attend a <a href="https://researchcomputing.princeton.edu/education/help-sessions">help session</a>.

## Acknowledgements

Kyle Felker has made improvements to this page.
