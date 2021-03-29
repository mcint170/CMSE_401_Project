# PyTorch Classification Example

[`PyTorch`](https://pytorch.org/) is a popular software/package for 
tensor computation and construction of Neural Networks. It is a pacakge for python, 
as well as C++. The software is also
[open source](https://github.com/pytorch/pytorch) and has a rich community 
that has a plethora of tutorials on how to use many of the features. One of 
`PyTorch`'s strengths is that it can utilize GPUs to perform calculations. 
This allows for significant speed up in training and computation. One thing 
that is interesting is the code is "not a Python binding into a monolithic 
C++ framework. It is built to be deeply integrated into Python." `PyTorch` 
also has many libraries such as [`torchaudio`](https://pytorch.org/audio/stable/index.html) 
for audio, [`torchtext`](https://pytorch.org/text/stable/index.html) for text, 
[`torchvision`](https://pytorch.org/vision/stable/index.html) for computer vision, 
[`TorchElastic`](https://pytorch.org/elastic/0.2.1/index.html) for running on 
changing environments, [`TorchServe`](https://pytorch.org/serve/) for serving 
`PyTorch` models. This package is great for projects that require large amounts
of computation through neural networks, such as neural networks that take in images. 

## ANACONDA SET UP:

In order to run this example easily, it is suggested that Anaconda 3 with python 
version 3.6 or higer is used. With using Anaconda, since many python pacakges
are already installed and conda is pre-installed, the setup will be less intensive.

If you already have Anaconda 3 with python version 3.6 or higher, you can skip
to Virtual Enviroment Setup.

Below are instructions on how to install Anaconda 3 with python version 3.8 on
the HPCC. We will also set up a new virtual enviroment with the packages we 
will need to install to get our example working. Please note this would probably
work best if you can keep this README.md open and also have access to a terminal
in another window.

### INSTALL Anaconda 3 w/ Python 3.8:

The following was created on 3/22/2021. It is possbile at the time you are reading
this the Anaconda version has updated. If this is the case, visit Anconda's website
and use the link to the newest Anconda version in place of the link below. Also
make sure to change the bash command to run the appropiate script.

1. In your home directory (~), or wherever you would like,
you will want to download the installation script from Anaconda's website. 
This can easily be done with this command.

```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
```

2. Next we need to run the installation script. To do this makes sure you are
in the same directory as the file we just downloaded. Then run the following.

```bash
bash Anaconda3-2020.11-Linux-x86_64.sh
```

3. The installation process will ask you a few things

4. Read through the License agreement by pressing/holding `ENTER`

5. Type "yes" to accept the terms

6. You will then be asked where you would like to install anaconda 3. 
Press "Enter" to save it in `~/anaconda3`, or if you have another location 
you would like to save it, you can enter this here. But note this guide 
assumes this is installed in the home directory. You may neeed to modify 
directions later based on where you save anaconda3.

7. Once finished installing, it will ask if you would like the installer to 
initalize Anaconda3. For using this as a module, you are going to want 
to say "No". However, if you want Anaconda3 always avalible to you anytime, 
you can say "yes" to this.

8. You can now delete or move the `Anaconda3-2020.11-Linux-x86.sh` file to 
wherever you would like.

9. Now you can load in Anaconda 3 with the following command.

```bash
module load Anaconda/3
```

10. You can test that the installation worked properly if the follwing command 
returns `conda 4.9.2` or something equivalent.

```bash
conda --version
```

### Virtual Enviroment Setup: 

From the last section, make sure you Anaconda 3 is active. If you followed the
process above this was done with 

```bash
module load Anaconda/3
conda --version
```

If a version number was given you are good to go.

There are two routes to go from here. Either you can load the enviroment included
or you can create a new enviroment

**Load the Enviroment from yml**

Loading the enviroment allows you to use the same exact enviroment I used to 
run the code

1. In the terminal navigate to the same folder this README.md is in 
(`/pytorch_classifier`).

2. You can create the enviroment from the `pytorch_classifier.yml` file with
```bash
conda env create -f pytorch_classifier.yml
```

3. To activate the enviroment we can use
```bash
conda activate pytorch_classifier
```

**Create new Enviroment**

Setting up a new virtual enviroment will help keep your base Anaconda enviroment 
clean so that way you can have seperate enviroment where we can install our 
packages and not have to worry about them conflicting in the future with other
packages.

1. In the terminal navigate to the same folder this README.md is in 
(`/pytorch_classifier`).

2. Now we can create our new enviroment with this command.
```bash
conda create -n pytorch_classifier pytorch torchvision requests cudatoolkit=10.2 -c pytorch
```

3. We are now able to activate our enviroment with 
```bash
conda activate pytorch_classifier
```

**Notes**

Note that when you are done and want to get out of this enviroment you can use
```bash
conda deactivate
```

Also if in the future you would like to delete the enviroment you can use
```bash
conda env remove -n pytorch_classifier
```

## Module Setup

In order to take full advantage of this example, it is suggested to use a
developer node with GPU access (`dev-intel16-k80`,
`dev-amd20-v100` ---- Note that `dev-intel14-k20` does not work with this
example).

If Anaconda 3 is installed with the appropiate package and enviroment from above.
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

The code comes from [this tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)
I simply just combined it into one script and implemented GPU usage.

To run the code is simple. I have included a makefile to streamline the process.

In order to start the program simply run
```
make
```

This should autodetect if you are on a node with GPU and run the appropiate
version of the code.

The code will train a deep neural network with 10000 images.
The network will then be tested with 10000 images.
The results of the classification are then printed at the end.

The program will produce the following:
 * `/data` --  folder containing data for training and testing model
 * `training.png` -- Sample grid of training images
 * `testing.png` -- Sample grid of testing images 
 * `cifar_net.pth` -- Trained Neural Network 

These files can easily be deleted with this command
```
make clean
```

If you would like to run this as a HPCC job, you can use the command
```
sbatch classifier.sb
```

This concludes this README, below is references

# References

Code:  
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py  

Conda Enviroments:  
https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533#e814
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#removing-an-environment


