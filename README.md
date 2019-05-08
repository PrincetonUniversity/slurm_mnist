# Installing and Running TensorFlow on the HPC Clusters

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
module load anaconda3
conda create -n tf-gpu matplotlib tensorflow-gpu
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

Then from the `slurm_mnist` directory run:

```
sbatch mnist.cmd
```

This will request a GPU, 5 minutes of computing time, and queue the job. You
should receive a job number and can check if your job is running or queued
via `squeue -u yourusername`.

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
conda create -n tf-cpu matplotlib tensorflow
```

Please send questions/issues to cses@princeton.edu.
