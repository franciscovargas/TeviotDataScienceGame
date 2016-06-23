# TeviotDataScienceGame
This repository is for the edinburgh university data science game entry.

## Installation Guide (With OpenBLAS)

### OpenBLAS/numpy install

There is a script provided to use all of ashburys cores:

    ssh student.compute
    chmod a+x install.sh
    ./install.sh
    export OMP_NUM_THREADS=40
    . cord3/core3/bin/acitivate

This creates a venv valled core3 inside a folder called cord3. 

### Project install

After installing openblass/numpy install this project such that the imports work do the following command
on DICE machines (within the root directory of this project)

    # inside virtual env locally with BLAS numpy preinstals
    pip install -e .
    # or globally in dice but without BLAS/Numpy core config facility
    pip install --user -e .

All this within the virtualenv you are now ready to train the current keras models in train_scripts using all 
of ashburies cores. The only things you need to do when logging in are activating the venv ```. core3/bin/acitivate``` and setting the enviroment variable```OMP_NUM_THREADS=40``` .
