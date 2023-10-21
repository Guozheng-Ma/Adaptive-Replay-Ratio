# **Revisiting Plasticity in Visual Reinforcement Learning: Data, Modules and Training Stages**

This code is for the paper titled ***Revisiting Plasticity in Visual Reinforcement Learning: Data, Modules and Training Stages***.

## Setup

Install [MuJoCo](http://www.mujoco.org/) if it is not already installed:

- Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html).
- Download MuJoCo binaries [here](https://www.roboti.us/index.html).
- Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.
- Use the env variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo license key path and the MuJoCo directory path.
- Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

Install the following libraries:

```
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Install dependencies:

```
conda env create -f conda_env.yml
conda activate drqv2
```

## Training Agent

Train DrQ-v2 agent with **Adaptive Replay Ratio(Our method)**:

```
bash train_adapt_rr.sh
```

Train DrQ-v2 agent, you can customize replay ratio by adjusting *replay ratio* and *agent.update_every_steps*:

```
bash train.sh
```

