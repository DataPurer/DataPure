# DataPure
This repository contains the PyTorch implementation of "DataPure: Purifying Poisoned Dataset Using Diffusion Models".

To reproduce the experiments, follow these steps:

1. Run *_attack.py under the 'attack/' directory to train the backdoor model.
2. After training, run the corresponding *_.defense.py to use DataPure for backdoor defense. 

To set up the required environment, please refer to [BackdoorBench v1](https://github.com/SCLBD/BackdoorBench/tree/v1) and [Pytorch_diffusion](https://github.com/pesser/pytorch_diffusion)

We are currently working on reorganizing the code structure and plan to commit DataPure to [BackdoorBench v2](https://github.com/SCLBD/BackdoorBench) in the near future.
