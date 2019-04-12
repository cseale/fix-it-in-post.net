#!/bin/bash
sudo -u ec2-user -i << 'EOF'

# Lifecycle Configuration for Notebook Instance
export WORKING_DIR="/home/ec2-user/SageMaker/fix-it-in-post.net"

# use bash
export SHELL="/bin/bash"

# use pytorch environment
source activate pytorch_p36

# install tensorflow
pip install tensorflow

# install tensorboard 
pip install tensorboard
pip install tensorboardx 

# automate this activation to work everytime you open a terminal
echo "source activate pytorch_p36" >> ~/.bashrc 
echo "cd $WORKING_DIR" >> ~/.bashrc 
