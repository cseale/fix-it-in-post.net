# load important libaries

# activate pytorch
source activate pytorch_p36

# automate this activation to work everytime you open a terminal
echo 'source activate pytorch_p36' >> ~/.bashrc 

# install tensorflow
# conda install --yes tensorflow

# install tensorboard 
# conda install --yes tensorboard
conda install -y -c conda-forge tensorboardx 
