/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <faiss/IndexFlat.h>

// 64-bit int
using idx_t = faiss::idx_t;



#include <vector>
#include <algorithm>

// Function to perform argsort on a vector
template <typename T>
std::vector<size_t> argsort(const std::vector<T>& vec) {
    // Initialize indices vector with the indices [0, 1, ..., vec.size() - 1]
    std::vector<size_t> indices(vec.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Sort indices vector based on values in the original vector
    std::sort(indices.begin(), indices.end(), [&vec](size_t i, size_t j) {
        return vec[i] < vec[j];
    });

    return indices;
}



int main() {

    int d = 64;      // dimension
    int nb = 100; // database size
    int nq = 10;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];
    float* r_qua = new float[nb];

    for (int i = 0; i < nb; i++) {
        r_qua[i] = distrib(rng); 
        if (i < 10) { 
            printf("r_qua = %f \n", r_qua[i]); 
        }
    }


    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }






    // faiss::IndexFlatIP index(d); // call constructor
    faiss::IndexFlatL2 index(d); // call constructor
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");
    // index.add(nb, xb); // add vectors to the index
    index.add(nb, xb, r_qua);
    printf("ntotal = %zd\n", index.ntotal);
    printf("%ld ", index.qua_size);

    
    /* Check reconstruct vector*/

    float* resconstruc_vector = new float[nb]; 

    // index.sa_decode(1, index.codes.data() + 0 * index.code_size, resconstruc_vector_2);
    index.sa_qua_decode(nb, index.qualities.data(), resconstruc_vector);

    for (int i = 0; i < 10; i++) {
        printf("%f %f\n", resconstruc_vector[i], r_qua[i]);
    }


    std::vector<float> distance_ip; 
    distance_ip.resize(nb); 

    for (int i = 0; i < nb; i++){
        float ip = 0.0;

        for (int j = 0; j < d; j++) {
            ip += fabs(xb[1 * d + j] - xb[i * d + j]) * fabs(xb[1 * d + j] - xb[i * d + j]); 
        }
        distance_ip[i] = ip; 
    }

    std::vector<size_t> sorted_indices = argsort(distance_ip);

    // Print sorted indices
    for (size_t i : sorted_indices) {
        if (resconstruc_vector[i] >= 0 && resconstruc_vector[i] <= 0.2){
            std::cout << "( " << distance_ip[i] << ", " << resconstruc_vector[i] << ",  " << i << " ) " << " ";
        }

    }
    std::cout << std::endl;

    int k = 1;

    { // sanity check: search 5 first vectors of xb
        idx_t* I = new idx_t[k * 5];
        float* D = new float[k * 5];

        index.search(5, xb, k, D, I);

        // print results
        printf("I=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++) {
                printf("%5zd ", I[i * k + j]);
                printf("%5g ", resconstruc_vector[(I[i * k + j])]);
            }
            
            printf("\n");
        }

        printf("D=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    // printf("%ld", index.get_qualities().size());
    // if (int(index.ntotal * index.qua_size) == index.qualities.size()){
    //     printf("hello");
    // }
    // return 0;
    { // sanity check: search 5 first vectors of xb
        idx_t* I = new idx_t[k * 5];
        float* D = new float[k * 5];

        index.search_with_quality(5, xb, k, 0, 0.2, D, I);

        // print results
        printf("I=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++){
                printf("%5zd ", I[i * k + j]);
                printf("%5g ", resconstruc_vector[(I[i * k + j])]);
            }
                
            printf("\n");
        }

        printf("D=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    // { // search xq
    //     idx_t* I = new idx_t[k * nq];
    //     float* D = new float[k * nq];

    //     index.search(nq, xq, k, D, I);

    //     // print results
    //     printf("I (5 first results)=\n");
    //     for (int i = 0; i < 5; i++) {
    //         for (int j = 0; j < k; j++)
    //             printf("%5zd ", I[i * k + j]);
    //         printf("\n");
    //     }

    //     printf("I (5 last results)=\n");
    //     for (int i = nq - 5; i < nq; i++) {
    //         for (int j = 0; j < k; j++)
    //             printf("%5zd ", I[i * k + j]);
    //         printf("\n");
    //     }

    //     delete[] I;
    //     delete[] D;
    // }

    delete[] xb;
    delete[] xq;

    return 0;
}
