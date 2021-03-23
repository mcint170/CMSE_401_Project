# PyTorch Classification Example

PyTorch is a python package that specializes in tensor computation and
construction of Neural Networks.

## ANACONDA SET UP:

In order to run this example easily, it is suggested that Anaconda 3 with python 
version 3.6 or higer is used. With using Anaconda, since many python pacakges
are already installed and conda is pre-installed, the setup will be less intensive.

If you already have Anaconda 3 with python version 3.6 or higher, you can skip
to Kernel Setup section if you would like to keep any installations

Below are instructions on how to install Anaconda 3 with python version 3.8 on
the HPCC. We will also set up a new virtual enviroment with the packages we 
will need to install to get our example working. Please note this would probably
work best if you can keep this README.md open and also have access to a terminal.

### INSTALL Anaconda 3 w/ Python 3.8:

The following was created on 3/22/2021. It is possbile at the time you are reading
this the Anaconda version has updated. If this is the case, visit Anconda's website
and use the link to the newest Anconda version in place of the link below. Also
make sure to change the bash command to run the appropiate script.

1. In your home directory (~), or wherever you would like,
you will want to download the installation script from Anaconda's website. 
This can easily be done with this command.
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
```

2. Next we need to run the installation script. To do this makes sure you are
in the same directory as the file we just downloaded. Then run the following.
```
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
```
module load Anaconda/3
```

10. You can test that the installation worked properly if the follwing command 
returns `conda 4.9.2` or something equivalent.
```
conda --version
```

### Virtual Enviroment Setup: 

There are two routes to go here. Either you can load the enviroment included
or you can create a new enviroment

**Load the Enviroment from yml**

Loading the enviroment allows you to use the same exact enviroment I used to 
run the code

1. In the terminal navigate to the same folder this README.md is in 
(`/pytorch_classifier`).

2. You can create the enviroment from the `pytorch_classifier.yml` file with
```
conda env create -f pytorch_classifier.yml
```

3. To activate the enviroment we can use
```
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
```
conda create -n pytorch_classifier pytorch torchvision matplotlib requests ipykernel cudatoolkit=10.2 -c pytorch
```

3. We are now able to activate our enviroment with 
```
conda activate pytorch_classifier
```

**Notes**

Note that when you are done and want to get out of this enviroment you can use
```
conda deactivate
```

Also if in the future you would like to delete the enviroment you can use
```
conda env remove -n pytorch_classifier
```

## Module Setup

In order to take full advantage of this example, it is suggested to use a
developer node with GPU access (`dev-intel14-k20`, `dev-intel16-k80`,
`dev-amd20-v100`).

If Anaconda 3 is installed with the appropiate package and enviroment from above.
We can load Anaconda 3 with `pytorch_classifier` enviroment as
```
module load Anaconda/3
conda activate pytorch_classifier
```

Next, we need to load the appropiate modules for use of CUDA
```
module load GCC/8.3.0
module load CUDA/10.2.89
```

## Running the Code


