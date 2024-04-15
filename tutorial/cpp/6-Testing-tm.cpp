#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/invlists/InvertedLists.h>

using idx_t = faiss::idx_t;



int main(){

    int d = 64;      // dimension
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

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFScalarQuantizer index(&quantizer, d, nlist, true, faiss::ScalarQuantizer::QT_fp16);
    // faiss::IndexScalarQuantizer index(d, true, faiss::ScalarQuantizer::QT_fp16);

    index.train(nb, xb);
    // index.add(nb, xb);
    index.add(nb, xb, r_qua);
    index.make_direct_map(true);


    {
        float* resconstruct_vector = new float[d]; 
        float* qua_reconstruct = new float[10]; 

        // index.reconstruct_from_offset(0, 0, resconstruct_vector);
        for(int i = 0; i < 10; i++) {
            index.reconstruct(i, resconstruct_vector);
            index.reconstruct_qua(i, qua_reconstruct);

            // printf("%ld\n", index.invlists->code_size);

            // for (int j = 0; j < 5; j++){
            //     printf("%f %f \n", resconstruct_vector[j], xb[i*d + j]);
            // }
            printf("\n");
            printf("%f %f \n", qua_reconstruct[0], r_qua[i]);
            printf("\n");

        }
    }

    {
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];
        index.nprobe = 10; 
        index.search(nq, xq, k, D, I);
        printf("I=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd %lf ", I[i * k + j]), D[i * k + j];
            printf("\n");
        }

        index.search_with_quality(nq, xq, k, 0, 0.5, D, I);

        printf("I=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++){
                // index.reconstruct_qua(I[i * k + j], qua_reconstruct);
                printf("%5zd %f ", I[i * k + j], r_qua[(I[i*k +j])%10000]);
            }
            printf("\n");
        }
    }

    printf("Hello world!!!\n");
    return 0;
    
}