mkdir cord3
cd cord3
virtualenv core3
. core3/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install ipython
pip install notebook
OBDir="`pwd`"/"OpenBLAS"
git clone git://github.com/xianyi/OpenBLAS
cd OpenBLAS
make
make PREFIX=$OBDir install
cd ..
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OBDir/lib
wget https://sourceforge.net/projects/numpy/files/NumPy/1.11.0/numpy-1.11.0.zip
unzip numpy-1.11.0.zip
cd numpy-1.11.0
echo "[openblas]" >> site.cfg
echo "library_dirs = $OBDir/lib" >> site.cfg
echo "include_dirs = $OBDir/include" >> site.cfg
python setup.py install
pip install scipy
pip install -U matplotlib 
pip install argparse nose
pip install cython


