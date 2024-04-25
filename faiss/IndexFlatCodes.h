/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/Index.h>
#include <faiss/impl/DistanceComputer.h>
#include <vector>

namespace faiss {

struct CodePacker;

/** Index that encodes all vectors as fixed-size codes (size code_size). Storage
 * is in the codes vector */
struct IndexFlatCodes : Index {
    size_t code_size;

    static const size_t qua_size = sizeof(float); // size of quality 

    /// encoded dataset, size ntotal * code_size
    std::vector<uint8_t> codes;
    
    /// encoded quality, size ntotal * sizeof(float)
    std::vector<uint8_t> qualities;
    bool include_quality = false; 

    IndexFlatCodes();

    IndexFlatCodes(size_t code_size, idx_t d, MetricType metric = METRIC_L2);
    IndexFlatCodes(size_t code_size, idx_t d, bool include_quality_in, MetricType metric = METRIC_L2);

    /// default add uses sa_encode
    void add(idx_t n, const float* x) override;
    void add_with_quality(idx_t n, const float* x, const float* r_qua) override;

    bool get_include_quality();
    void set_include_quality();

    void reset() override;

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reconstruct_qua_n(idx_t i0, idx_t ni, float* qua_recons) const override;

    void reconstruct_qua(idx_t key, float* qua_recons) const override;

    size_t sa_code_size() const override;

    size_t sa_qua_code_size() const override;

    /** remove some ids. NB that because of the structure of the
     * index, the semantics of this operation are
     * different from the usual ones: the new ids are shifted */
    size_t remove_ids(const IDSelector& sel) override;

    /** a FlatCodesDistanceComputer offers a distance_to_code method */
    virtual FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const;

    DistanceComputer* get_distance_computer() const override {
        return get_FlatCodesDistanceComputer();
    }

    // returns a new instance of a CodePacker
    CodePacker* get_CodePacker() const;

    void check_compatible_for_merge(const Index& otherIndex) const override;

    virtual void merge_from(Index& otherIndex, idx_t add_id = 0) override;

    // permute_entries. perm of size ntotal maps new to old positions
    void permute_entries(const idx_t* perm);
};

} // namespace faiss
