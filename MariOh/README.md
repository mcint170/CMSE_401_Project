**Disclaimer:**
Please note, that most of this folders content, `MariOh/`, is from [here](https://github.com/yuansongFeng/MadMario/).
For example most of this README was not written by me.

Credit must be given to Yuansong Feng, Suraj Subramanian, Howard Wang, and Steven Guo.

My edits to any of the files will include a tag **(BM)** to indicate that I altered
that line/surrounding content. If a file was completely written by me I will include it in the list
below.

Files I created:
* Mario.sb

Also any instructions specific to the HPCC should be assumed to be written by 
myself.


# MadMario
PyTorch [official tutorial](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) to build an AI-powered Mario.

## Set Up
(Recommendation for X11 forwarding) **(BM)**  
* Create an Interctive Desktop on HPCC's [OnDemand](https://ondemand.hpcc.msu.edu/) for 1 hour,
with 1 core, 15 Gigabytes, Node type either `intel16` or `amd20`,and request 1 GPU. This will be useful later because we can use the already-baked-in
X11 forwarding to visualize how well our simulation did!
* Once logged into the desktop you will want to start the terminal by clicking 
the icon on the top-left of the screen, `System Tools`, and finally `Terminal`
* Run the command `nvida-smi` to make sure the graphics card is NOT `K20`, the K20
graphics card does not work with pytorch unfortunetly.
(End Recomendation)

1. Have Anaconda 3 installed. If you do not have Anaconda installed, see the [README](../pytorch_classifier/README.md) file
in `../pytorch_classifier` and follow the instructions under 
**INSTALL Anaconda 3 w/ Python 3.8**. **(BM)**
3. (If using instructions in Part 1, or do not have conda activated at startup)
The module `Anaconda /3` will need to be loaded with **(BM)**
```bash
module load Anaconda/3
```
2. Install dependencies with `environment.yml`
```bash
conda env create -f environment.yml
```
Check the new environment *mario* is [created successfully](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

3. Activate *mario* enviroment
```bash
conda activate mario
```


## Running Mario (BM)
There are two folders on this respository  

**single_GPU/**  
This is the original code, that only uses 1 GPU

**mulit_GPU/**
This is my modified code, that uses more than 1 GPU

You can navigate to either folder and follow the instructions below. They function
identically besides the number of GPU trained with.

To manually start the **learning** process for Mario we can run the following,
```
make test
```
This starts the *double Q-learning* for 10 epochs and logs key training metrics to `checkpoints`. 
In addition, a copy of `MarioNet` and current exploration rate will be saved. We will use the saved checkpoint
to visualize the machiene playing mario.

A single GPU will automatically be used if available.

To **visualize** our Mario we can run,
```
make visual
```
This visualizes Mario playing the game in a window (why we needed X11). Performance metrics 
also will be logged to a new folder under `checkpoints`. You can change the `checkpoint` variable, 
[FIXME]e.g. `checkpoints/2021-06-06T22-00-00`, in `Mario.load()` to check a specific timestamp if you 
would like. But it currently setup to use the checkpoint from the full trained model. 

## Project Structure 
This folder: **(BM)**  

**tutorial/**
Interactive tutorial with extensive explanation and feedback. Run it on [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true).

**enviroment.yml**
Used to construct working enviroment to run code. Was altered to work on HPCC **(BM)**

In both `single_GPU/` and `multi_GPU` **(BM)**  

**main.py**
Main loop between Environment and Mario

**agent.py**
Define how the agent collects experiences, makes actions given observations and updates the action policy.

**wrappers.py**
Environment pre-processing logics, including observation resizing, rgb to grayscale, etc.

**neural.py**
Define Q-value estimators backed by a convolution neural network.

**metrics.py**
Define a `MetricLogger` that helps track training/evaluation performance.

## Key Metrics

- Episode: current episode
- Step: total number of steps Mario played
- Epsilon: current exploration rate
- MeanReward: moving average of episode reward in past 100 episodes
- MeanLength: moving average of episode length in past 100 episodes
- MeanLoss: moving average of step loss in past 100 episodes
- MeanQValue: moving average of step Q value (predicted) in past 100 episodes

## Pre-trained

Checkpoint for a trained Mario: https://drive.google.com/file/d/1RRwhSMUrpBBRyAsfHLPGt1rlYFoiuus2/view?usp=sharing

## Resources

Deep Reinforcement Learning with Double Q-learning, Hado V. Hasselt et al, NIPS 2015: https://arxiv.org/abs/1509.06461

OpenAI Spinning Up tutorial: https://spinningup.openai.com/en/latest/

Reinforcement Learning: An Introduction, Richard S. Sutton et al. https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf

super-mario-reinforcement-learning, GitHub: https://github.com/sebastianheinz/super-mario-reinforcement-learning

Deep Reinforcement Learning Doesn't Work Yet: https://www.alexirpan.com/2018/02/14/rl-hard.html
