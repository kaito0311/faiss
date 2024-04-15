/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_SCALAR_QUANTIZER_H
#define FAISS_INDEX_SCALAR_QUANTIZER_H

#include <stdint.h>
#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

/**
 * Flat index built on a scalar quantizer.
 */
struct IndexScalarQuantizer : IndexFlatCodes {
    /// Used to encode the vectors
    ScalarQuantizer sq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexScalarQuantizer(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2);

    IndexScalarQuantizer();

    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void boundary_search(
            idx_t n,
            const float* x,
            idx_t k,
            const float lower,
            const float upper,
            const float duplicate_thr,
            const bool rm_duplicate,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void search_with_quality(
            idx_t n, 
            const float* x, 
            idx_t k, 
            const float lower_quality,
            const float upper_quality, 
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    /* standalone codec interface */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_qua_encode(idx_t n, const float* r_qua, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    void sa_qua_decode(idx_t n, const uint8_t* bytes, float* r_qua) const override;
};

/** An IVF implementation where the components of the residuals are
 * encoded with a scalar quantizer. All distance computations
 * are asymmetric, so the encoded vectors are decoded and approximate
 * distances are computed.
 */

struct IndexIVFScalarQuantizer : IndexIVF {
    ScalarQuantizer sq;

    IndexIVFScalarQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            bool by_residual = true);

    IndexIVFScalarQuantizer();

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    idx_t train_encoder_num_vectors() const override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx) override;

    void add_core(
            idx_t n,
            const float* x,
            const float* r_qua,
            const idx_t* xids,
            const idx_t* precomputed_idx) override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    void reconstruct_qua_from_offset(int64_t list_no, int64_t offset, float* qua_recons)
            const override;
            
    /* standalone codec interface */
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

} // namespace faiss

#endif
