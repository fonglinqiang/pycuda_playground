# pycuda_playground
Most codes are pushed by lqfong9 (my work github account)

This repo is meant for my personal growth and interest in learning CUDA through PyCUDA

# Setup Environment
- This is setup is for ubuntu-20.04.
- Setup Docker and Nvida container toolkit: [instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- clone the TensorRT repo at your directory
```shell
git clone https://github.com/NVIDIA/TensorRT
bash ./docker/build.sh --file docker/ubuntu-20.04.Dockerfile --tag tensorrt-ubuntu20.04-cuda12.0
```
- After the docker image is built successfully, change directory to this repo
```shell
cd /path/to/pycuda_playground
docker run -d --gpus=all -it -v $PWD:/pycuda_playground --net=host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --name pycuda_playground tensorrt-ubuntu20.04-cuda12.0
```
Open VS Code, install `Dev Containers` extension and open the pycuda_playground workspace using `attached to running container` option