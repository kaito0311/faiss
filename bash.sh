# Create anaconda environment
conda create -n imint python=3.8
conda activate imint
conda install mkl mkl_fft
# Install custom FAISS
git clone https://github.com/NNDam/faiss.git
cd faiss
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_OPT_LEVEL=avx2 -DBUILD_SHARED_LIBS=ON
make -C build -j8 faiss
make -C build -j8 swigfaiss
cd build/faiss/python && python setup.py install


# Running an example 
## Create an new file such as 6-Testing-tm.cpp 
## Run make to rebuild, if file already exist, skipp this step 
make -C build -j8 faiss
## Run make to compile 
make -C build 6-Testing-tm
## Run file 
./build/tutorial/cpp/6-Testing-tm 