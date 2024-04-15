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
make -C build -j8 faiss faiss_avx2
## Run make to compile 
make -C build 6-Testing-tm
make -C build 2-IVFFlat
## Run file 
./build/tutorial/cpp/6-Testing-tm 
make -C build 1-Flat && ./build/tutorial/cpp/1-Flat
make -C build 2-IVFFlat && ./build/tutorial/cpp/2-IVFFlat
make -C build 6-Testing-tm && ./build/tutorial/cpp/6-Testing-tm
make -C build 7-Read_write_index && ./build/tutorial/cpp/7-Read_write_index
