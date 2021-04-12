# Timing Study

This folder contains the code and scripts used to performed the timing
study on the classification example. For this example I changed the `batch_size`
and the number of epochs run to make this problem more suitable for parallelization. 
With a small `batch_size` the GPU parallelization actaually performs worse (becuase
of overhead). For the time comparsion, I will compare the improvement using 4 
GPU can bring, per `batch_size` of this classifier.

Note that full installation and setup can be found in `../pytorch_classifier/README.md`.
Since there is quite a bit of information I decieded to keep that sepearte from this
`README.md` file. However, here is the minimum things you will need.

* Anconda 3 with python 3.8 installed
* `pytorch_classifier` enviroment created

## Module Setup

In order to run the code without job submission, it is suggested to use a
developer node with GPU access (`dev-intel16-k80`,
`dev-amd20-v100` ---- Note that `dev-intel14-k20` does not work with this
example).

If Anaconda 3 is installed with the appropiate package and enviroment..
We can load Anaconda 3 with `pytorch_classifier` enviroment as
```bash
module load Anaconda/3
conda activate pytorch_classifier
```

Next, we need to load the appropiate modules for use of CUDA
```bash
module load GCC/8.3.0
module load CUDA/10.2.89
```

## Running the Code

The base code comes from [this tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)
I simply just combined it into one script and implemented GPU usage.

I also have now implemented the use of multiple GPU usage as well.

To run the code is simple. I have included a makefile to streamline the process.

In order to start the program with 1 GPU or 1 CPU only simply run
```
make single
```

To start the program with multiple GPU. (If on `dev-intel16-k80` this will be 8 GPUs,
if on `dev-amd20-v100` this will be 4 GPUs)

```
make multi
```

The code will train a deep neural network with 10000 images for 2 epochs.
The network will then be tested with 10000 images.
The results of the classification are then printed at the end.

*Note: `{VERSION}` will either be `single` or `multi`.*
The program will produce the following:
 * `/data` --  folder containing data for training and testing model
 * `training_{VERSION}.png` -- Sample grid of training images
 * `testing_{VERISON}.png` -- Sample grid of testing images 
 * `cifar_net_{VERSION.pth` -- Trained Neural Network 

These files can easily be deleted with this command
```
make clean
```

*Note: `{VERSION}` will either be `single` or `multi`.*
If you would like to run the timing test of these as a HPCC job, you can use the command.
```
sbatch {VERSION}_classifier.sb
```

This will run the classifier with either 1 or 4 GPU, depending on verison, for 20 epochs.

To see the results of this timing study, please see `../Classifier_Part_2.ipynb`

# References

Code:  
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py  

Conda Enviroments:  
https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533#e814
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#removing-an-environment

Parallelization:
https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#create-model-and-dataparallel

