#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/invlists/InvertedLists.h>
#include <faiss/invlists/OnDiskInvertedLists.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/AutoTune.h>
// 64-bit int
using idx_t = faiss::idx_t;



#include <vector>
#include <algorithm>

int main() {



    int d = 512;      // dimension
    int nb = 10000; // database size
    int nq = 1000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];
    idx_t* ids = new idx_t[nb];
    float* r_qua = new float[nb];
    
    for (int i = 0; i < nb; i++) {
        r_qua[i] = distrib(rng); 
    }

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
        ids[i] = i; 
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    int nlist = 10;
    int k = 4;

    /* ==================== Search ondisk =======================*/

    // std::string filename = "tmp_fiass";

    // faiss::IndexFlatL2 quantizer(d);

    // faiss::IndexIVFFlat index(&quantizer, d, nlist);

    // faiss::OnDiskInvertedLists ivf(
    //                 index.nlist, index.code_size, filename.c_str());

    // index.replace_invlists(&ivf);

    // idx_t* I = new idx_t[k * nq];

    // float* D = new float[k * nq];

    // index.search(nq, xq, k, D, I);

    // faiss::Index* index = faiss::read_index("/home1/data/tanminh/dev_faiss/test_faiss/saved/ondisk/index_ivfsq.bin");
    // faiss::Index* index = faiss::read_index("/home1/data/tanminh/dev_faiss/test_faiss/saved/onram/index_ivfsq.bin",
    //                                         faiss::IO_FLAG_MMAP);
    // const faiss::IndexIVFScalarQuantizer* ivsc = dynamic_cast<const faiss::IndexIVFScalarQuantizer*>(index);
    // // index->add_with_ids(nb, xb, ids);
    // printf("[INFO] Read done\n");
    // index->add_with_ids_with_quality(nb, xb, r_qua, ids);
    // faiss::write_index(index, "./indexflat.bin");

    // faiss::Index* index2 = faiss::read_index("/home1/data/tanminh/dev_faiss/test_faiss/saved/ondisk/index_ivfsq.bin", 0);







    /* ===============================================================*/


    // faiss::Index* index = faiss::read_index("./index_ivf_sq_idmap.bin", 0);
    // const faiss::IndexIVFScalarQuantizer* idxf = dynamic_cast<const faiss::IndexIVFScalarQuantizer*>(index);
    // faiss::IndexScalarQuantizer index(d, true, faiss::ScalarQuantizer::QT_fp16);
    // faiss::IndexFlatL2 quantizer(d);
    // faiss::IndexIVFScalarQuantizer index_scalar(&quantizer, d, nlist, true, faiss::ScalarQuantizer::QT_fp16);
    // faiss::IndexIDMap index(&index_scalar);
    // // faiss::IndexFlatL2 quantizer(d); // the other index
    // // faiss::IndexIVFFlat index(&quantizer, d, nlist, true);
    // index.train(nb, xb);
    // index.add_with_ids(nb, xb, r_qua, ids);

    // /* Test index write*/ 
    // faiss::write_index(&index, "./index_ivf_sq_idmap.bin");
    // return 0; 

    // idx_t* I = new idx_t[k * nq];
    // float* D = new float[k * nq];
    // index->search(nq, xq, k, D, I);

    // printf("I=\n");
    // for (int i = nq - 5; i < nq; i++) {
    //     for (int j = 0; j < k; j++)
    //         printf("%5zd %f ", I[i * k + j], r_qua[(I[i*k +j])%10000]);
    //     printf("\n");
    // }

    // // idxf->nprobe = 10;
    // faiss::ParameterSpace pspace = faiss::ParameterSpace();
    // pspace.set_index_parameter(index, "nprobe", 10);

    // index->search_with_quality(nq, xq, k, 0, 0.5, D, I);
    // printf("I=\n");
    // for (int i = nq - 5; i < nq; i++) {
    //     for (int j = 0; j < k; j++){
    //         // index.reconstruct_qua(I[i * k + j], qua_reconstruct);
    //         printf("%5zd %f ", I[i * k + j], r_qua[(I[i*k +j])%10000]);
    //     }
    //     printf("\n");
    // }


    // return 0; 


    // /* Test index write*/ 
    // faiss::IndexFlatL2 index(d, true); // call constructor
    // index.add(nb, xb, r_qua);
    // faiss::write_index(&index, "./indexflat.bin");
    // return 0; 



}