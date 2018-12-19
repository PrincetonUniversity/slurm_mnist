# Using this example

This example is meant to be a repackage of one of the basic TensorFlow tutorials
for use on one of Princeton University's HPC clusters. It gives a basic recipe
for how to work around a few of the things that make adapting these a challenge.
And it will help to make sure you have a working GPU environment.

## First, read the actual tutorial
This adapts one exercise from the TensorFlow/Keras [basic classification tutorial](https://www.tensorflow.org/tutorials/keras/basic_classification).

## Clone the repo

First, clone the repo to your home directory using:

```
git clone https://github.com/bwhicks/slurm_mnist.git
```

This will get you the file in a folder called `slurmn_mnist`

## Make an appropriate conda environment

Getting GPU (and some nice MKL) support for Tensorflow is as easy as:

```
module load anaconda3
conda create -n tf-gpu matplotlib tensorflow-gpu
```

Once this command completes, as long as you have the `anaconda3` module loaded,
you'll have access to `conda` and can use it to access the Python
virtual environment you just created:

```
conda activate tf-gpu
```

## Verify that you have a working GPU enabled Tensorflow

There are no GPUs on the head node of any of the clusters with GPU capacity,
so you'll need to get one via the scheduler. You can do that interactively with:

```
salloc --gres=gpu:1 -t 00:05:00
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

Then from the `slurm_mnist` run:

```
sbatch mnist.cmd
```

This will request a GPU, 5 minutes of computing time, and queue the job. You
should receive a job number and can check if your job is running or queued
via `squeue -u yourusername`.

You'll also receive start and finish emails.

Once the job runs, you'll have a `slurm-xxxxx.out` file in the `slurm_mnist` with
tensorflow's messages, and the one expect output, an example graph of images
with tensorflow's best guess (and whether it was right) for what article of
clothing they might be.
