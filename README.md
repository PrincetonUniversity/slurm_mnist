# Installing and Running TensorFlow on the HPC Clusters

[TensorFlow](https://www.tensorflow.org) is a popular deep learning library for training artificial neural networks. The installation instructions depend on the version and cluster. This page covers version 2.x. Directions for TensorFlow 1.x are [here](version_1.x/README.md).

# Version 2.x

### TigerGPU or Adroit

TensorFlow 2.x is available on Anaconda Cloud:

```
$ module load anaconda3
$ conda create --name tf2-gpu tensorflow-gpu <package-2> <package-3> ... <package-N>
$ conda activate tf2-gpu
```

Be sure to include `conda activate tf2-gpu` and `#SBATCH --gres=gpu:1` in your Slurm script.

### TigerCPU, Della or Perseus

To install the CPU-only version of TensorFlow 2.0 follow the directions above except replace `tensorflow-gpu` with `tensorflow` and `tf2-gpu` with `tf2-cpu`. In the Slurm script below you must remove the gpu line (`#SBATCH --gres=gpu:1`) and be sure to use the environment name `tf2-cpu`. For tips on running on CPUs see [this page](cpu_only/README.md).

### Traverse

TensorFlow 2.x is available for IBM's Power architecture and NVIDIA GPUs on IBM's Conda channel, Watson Machine Learning Community Edition. Follow these installation directions:

```bash
$ module load anaconda3
$ conda create --name=tf2-gpu --channel https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda tensorflow2-gpu
$ conda activate tf2-gpu
```

Be sure to include `conda activate tf2-gpu` and `#SBATCH --gres=gpu:1` in your Slurm script.

## Example

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

Below is the TensorFlow script (`mnist_classify.py`) which trains a classifier on the MNIST data set:

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
#SBATCH --mem=4G                 # total memory per node (default is 4 GB per CPU-core)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3
conda activate tf2-gpu

srun python mnist_classify.py
```

## Multithreading

Even when using a GPU there are still operations that are carried out on the CPU. Some of these operations have been written to take advantage of multithreading. Try different values of `--cpus-per-task` to see if you get a speed-up:

```
#!/bin/bash
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=<T>      # cpu-cores per task (>1 if multi-threaded tasks)
```

On TigerGPU, there are seven CPU-cores for every one GPU. Try doing a set of runs where you vary `<T>` from 1 to 7 to find the optimal value.

## GPU Utilization

To see how effectively your job is using the GPU, after submitting the job run the following command:

```
$ squeue -u $USER
```

The rightmost column labeled "NODELIST(REASON)" gives the name of the node where your job is running. Connect to this node:

```
$ ssh tiger-iXXgYY
```

In the command above, you must replace XX and YY with the actual values (e.g., `ssh tiger-i19g1`). Once on the compute node run `watch -n 1 gpustat`. This will show you a percentage value indicating how effectively your code is using the GPU. The memory allocated to the GPU is also available. TensorFlow by default takes all available GPU memory. For this specific example you will see that only about 10% of the GPU cores are utilized. Given that a CNN is being trained on small images (i.e., 28x28 pixels) this is not surprising. You should repeat this analysis with your actual research code to ensure that the GPU is being utilized. For jobs that run for more than 10 minutes you can check utilization by looking at the [TigerGPU utilization dashboard](https://researchcomputing.princeton.edu/node/7171). See the bottom of that page for tips on improving utilization.

Type `Ctrl+C` to exit the `watch` command. Type `exit` to leave the compute node and return to the head node.

TensorFlow will run all possible operations on the GPU by default. However, if you request more than one GPU in your Slurm script TensorFlow will only use one and ignore the others unless your actively make the appropriate changes to your TensorFlow script. This is covered next.

## Distributed Training or Using Multiple GPUs

Most models can be trained on a single GPU. If you are effectively using the GPU as determined by the procedure above then you may consider running on multiple GPUs. In general this will lead to shorter training times but because more resources are required the queue time will increase. When using more resources for your job you should always do a scaling analysis. Here is an [example](https://github.com/PrincetonUniversity/gpu_programming_intro/tree/master/05_multithreaded_numpy) for CPUs (see table at the bottom).

Models that are built using `tf.keras` can be made to run on multiple GPUs quite easily. This is done by using a [data parallel](https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_keras) approach where a copy of the model is assiged to each GPU where it operates on a different mini-batch. TensorFlow offers ways to use multiple GPUs with the subclassing API as well (see [tf.distribute](https://www.tensorflow.org/api_docs/python/tf/distribute) and [tutorials](https://www.tensorflow.org/guide/distributed_training)).

TensorFlow offers an approach for using multiple GPUs on [multiple nodes](https://www.tensorflow.org/guide/distributed_training#multiworkermirroredstrategy). [Horovod](https://github.com/horovod/horovod) can also be used.

For hyperparameter tuning consider using a [job array](https://github.com/PrincetonUniversity/hpc_beginning_workshop/tree/master/06_slurm). This will allow you to run multiple jobs with one `sbatch` command. Each job within the array trains the network using a different set of parameters.

## Tensorboard

[Tensorboard](https://www.tensorflow.org/tensorboard/get_started) comes included in a Conda installation of TensorFlow. It can be used to view your graph, monitor training progress and more. It can be used on the head node of a cluster in non-intensive cases. It can be used intensively on Tigressdata. See this [page](https://researchcomputing.princeton.edu/sshX) for using graphics on the HPC clusters.

## Using PyCharm on TigerGPU

This video shows how to launch PyCharm on a TigerGPU compute node and use its debugger on a TensorFlow script:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=0XmZsfixAdw" target="_blank">
<img src="http://img.youtube.com/vi/0XmZsfixAdw/0.jpg" alt="PyCharm" width="480" height="270" border="0" /></a>

## Inference

Some TigerGPU users are making use of [TensorRT](https://developer.nvidia.com/tensorrt), an SDK for high-performance deep learning inference.

The Cascade Lake nodes on Della are capable of Intel [Vector Neural Net Instructions]((https://www.intel.ai/vnni-enables-inference/#gs.yz55z2)) (VNNI) (a.k.a. DL Boost). The idea is to cast the FLOAT32 weights of your trained model to the INT8 data type.

## Building from Source

At times it may be necessary to build TensorFlow [from source](https://www.tensorflow.org/install/source). The procedure below was found to work on Della:

```bash
# ssh della
$ module load anaconda3
# anaconda3 provides us with pip six numpy wheel setuptools mock
$ pip install -U --user keras_applications --no-deps pip install -U --user keras_preprocessing --no-deps

$ cd /tmp/bazel/
$ wget https://github.com/bazelbuild/bazel/releases/download/2.0.0/bazel-2.0.0-installer-linux-x86_64.sh
$ chmod +x bazel-2.0.0-installer-linux-x86_64.sh
$ ./bazel-2.0.0-installer-linux-x86_64.sh --prefix=/tmp/bazel export PATH=/tmp/bazel/bin:$PATH

$ cd ~/sw
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ ./configure  # took defaults or answered no
$ module load rh/devtoolset/7
$ CC=`which gcc` BAZEL_LINKLIBS=-l%:libstdc++.a bazel build --verbose_failures //tensorflow/tools/pip_package:build_pip_package
$ ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
$ pip install --user /tmp/tensorflow_pkg/tensorflow-2.1.0-cp37-cp37m-linux_x86_64.whl
$ cd
$ python
>>> import tensorflow as tf
```

## How to Learn TensorFlow

See the [tutorials](https://www.tensorflow.org/tutorials) and [guides](https://www.tensorflow.org/guide). See [examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples) on GitHub.

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
