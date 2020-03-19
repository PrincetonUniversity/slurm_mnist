# Version 1.x

Note that keras is not included in version 1.x of TensorFlow. See the second example below for TigerGPU for keras installation directions. If you are new to installing Python packages then see <a href="https://researchcomputing.princeton.edu/python">this page</a>&nbsp;before continuing.

### TigerGPU or Adroit

```
$ module load anaconda3
$ conda create --name tf-gpu tensorflow-gpu=1.15 <package-2> <package-3> ... <package-N>
$ conda activate tf-gpu
```

Or maybe you want a few additional packages like keras and matplotlib:

```
$ module load anaconda3
$ conda create --name tf-gpu tensorflow-gpu=1.15 keras matplotlib <package-4> <package-5> ... <package-N>
$ conda activate tf-gpu
```

Be sure to include `conda activate tf-gpu` and `#SBATCH --gres=gpu:1` in your Slurm script.

### Traverse

```
$ module load anaconda3
$ conda create --name=tf-gpu --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda tensorflow-gpu=1.15
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
$ cd slurm_mnist/version_1.x
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
you'll note that we load it in the Slurm script `job.slurm`),
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

Edit the line of `job.slurm` to remove the space in the `--mail-user` line
and add your Princeton NetID (or other email address)

Then from the `slurm_mnist/version_1.x` directory run:

```
$ sbatch job.slurm
```

This will request a GPU, 5 minutes of computing time, and queue the job. You
should receive a job number and can check if your job is running or queued
via `squeue -u $USER`.

You'll also receive start and finish emails.

Once the job runs, you'll have a `slurm-xxxxx.out` file in the `slurm_mnist` with
tensorflow's messages, and the one expected output, an example graph of images
with tensorflow's best guess (and whether it was right) for what article of
clothing they might be. Download it via `scp` (or connect using `ssh -Y` with an XWindows server and use `eog`) and take a look!
