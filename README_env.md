# Environment 
## System Requirements
The experiments are conducted on Ubuntu 16.04, with Python version 3.6.9, Driver Version: 430.64, Cuda 9.0.

## MuJoCo
[MuJoCo](http://mujoco.org/) is a physics engine for detailed, efficient rigid body simulations with contacts.
`mujoco-py` allows using MuJoCo from Python 3.

Note that the [MuJoCo](http://mujoco.org/) version used in this environment is: mujoco-py=2.0.2.9  
Follow the instrcution in [mujoco-py](https://github.com/openai/mujoco-py#install-mujoco)
1. Download the MuJoCo version 2.0 binaries for
   [Linux](https://roboti.us/download.html).
1. Extract the downloaded `mujoco200_linux.zip` directory into `~/.mujoco/mujoco200`.
1. Download the MuJoCo license file `mjkey.txt` [license](https://roboti.us/license.html) and put in directory `~/.mujoco`
1. `pip install -r requirement.txt`

## Troubleshooting
Please check the [main page](https://github.com/openai/mujoco-py#install-mujoco) and [issue](https://github.com/openai/mujoco-py/issues) of [mujoco-py](https://github.com/openai/mujoco-py#install-mujoco) for troubleshooting.

After installation, the `.bashrc` should contains something like: \
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dcpu/.mujoco/mujoco200/bin`  
`export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so`  
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-430`  
`export OMP_NUM_THREADS=1`  

and some package need to be installed before install mujoco by my experience:  
`sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3`



 