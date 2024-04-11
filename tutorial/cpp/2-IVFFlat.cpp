/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/invlists/InvertedLists.h>

using idx_t = faiss::idx_t;

int main() {

    // /// Test Invert list
    // faiss::ArrayInvertedLists il(100, sizeof(float), false);

    // printf("qualities size: %ld \n", il.qualities.size()); 
    // printf("has qualities : %ld \n", il.has_quality(2));

    


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

    faiss::IndexFlatL2 quantizer(d); // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist, true);
    // printf("%ld\n", index.invlists->get_include_quality());
    // printf("%ld\n", index.invlists->get_quality_size());
    // printf("%ld\n", index.invlists->get_codes_size());
    // printf("%ld\n", index.invlists->has_quality(0));
    // printf("%ld\n", index.invlists->nlist);
    index.train(nb, xb);
    index.make_direct_map(true);
    index.add(nb, xb, r_qua);
    for (int i = 0; i < nlist; i++){
        printf("list size of codes in cluster 0: %ld\n", index.invlists->list_size(i));
        printf("list size of qualities in cluster 0: %ld\n", index.invlists->quality_list_size(i));

    }

    // return 0; 

    // index.add(nb, xb);
    index.invlists->list_size(0);

    float* resconstruct_vector = new float[d]; 
    float* qua_reconstruct = new float[10]; 

    // index.reconstruct_from_offset(0, 0, resconstruct_vector);
    for(int i = 0; i < 10; i++) {
        index.reconstruct(i, resconstruct_vector);
        index.reconstruct_qua(i, qua_reconstruct);

        // printf("%ld\n", index.invlists->code_size);

        for (int j = 0; j < 5; j++){
            printf("%f %f \n", resconstruct_vector[j], xb[i*d + j]);
        }
        printf("\n");
        printf("%f %f \n", qua_reconstruct[0], r_qua[i]);
        printf("\n");

    }
    
    return 0; 

    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        index.search(nq, xq, k, D, I);

        printf("I=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        index.nprobe = 10;
        index.search(nq, xq, k, D, I);

        printf("I=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}
