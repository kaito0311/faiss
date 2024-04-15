/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexFlatCodes.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

IndexFlatCodes::IndexFlatCodes(size_t code_size, idx_t d, MetricType metric)
        : Index(d, metric), code_size(code_size) {}

IndexFlatCodes::IndexFlatCodes(size_t code_size, idx_t d, bool include_quality_in, MetricType metric)
        : Index(d, metric), code_size(code_size), include_quality(include_quality_in) {}


IndexFlatCodes::IndexFlatCodes() : code_size(0) {}

void IndexFlatCodes::add_with_quality(idx_t n, const float* x, const float* r_qua) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT_MSG(include_quality, "include_quality is current false");
    if (n == 0) {
        return;
    }
    codes.resize((ntotal + n) * code_size); /// ntotal * code_size = ntotal * d * sizeof(float)
    qualities.resize((ntotal + n) * qua_size); /// ntotal * 1 * sizeof(float)
    sa_encode(n, x, codes.data() + (ntotal * code_size));
    sa_qua_encode(n, r_qua, qualities.data() + (ntotal * qua_size)); /// move pointer to where need add 
    ntotal += n;
}

void IndexFlatCodes::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT_MSG(include_quality == false, "Index now has quality attribute, add vector without quality cause conflict");
    if (n == 0) {
        return;
    }
    codes.resize((ntotal + n) * code_size); /// ntotal * code_size = ntotal * d * sizeof(float)
    sa_encode(n, x, codes.data() + (ntotal * code_size));
    ntotal += n;
}

bool IndexFlatCodes::get_include_quality() {
    return include_quality;
}
void IndexFlatCodes::set_include_quality() {
    this->include_quality = true;
}

void IndexFlatCodes::reset() {
    codes.clear();
    if (get_include_quality()) {
        qualities.clear();
    }
    ntotal = 0;
}

size_t IndexFlatCodes::sa_code_size() const {
    return code_size;
}

size_t IndexFlatCodes::sa_qua_code_size() const {
    return qua_size;
}

size_t IndexFlatCodes::remove_ids(const IDSelector& sel) {
    idx_t j = 0;
    for (idx_t i = 0; i < ntotal; i++) {
        if (sel.is_member(i)) {
            // should be removed
        } else {
            if (i > j) {
                memmove(&codes[code_size * j],
                        &codes[code_size * i],
                        code_size);

                if (get_include_quality()){
                    memmove(&qualities[qua_size * j],
                            &qualities[qua_size * i],
                            qua_size);
                }
            }
            j++;
        }
    }
    size_t nremove = ntotal - j;
    if (nremove > 0) {
        ntotal = j;
        codes.resize(ntotal * code_size);

        if (get_include_quality()){
            qualities.resize(ntotal * qua_size);
        }
    }
    return nremove;
}

void IndexFlatCodes::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
    sa_decode(ni, codes.data() + i0 * code_size, recons);
}

void IndexFlatCodes::reconstruct(idx_t key, float* recons) const {
    reconstruct_n(key, 1, recons);
}

void IndexFlatCodes::reconstruct_qua_n(idx_t i0, idx_t ni, float* qua_recons) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
    FAISS_THROW_IF_NOT(include_quality); 
    sa_qua_decode(ni, qualities.data() + i0 * qua_size, qua_recons);
}

void IndexFlatCodes::reconstruct_qua(idx_t key, float* qua_recons) const { 
    reconstruct_qua_n(key, 1, qua_recons);
}

FlatCodesDistanceComputer* IndexFlatCodes::get_FlatCodesDistanceComputer()
        const {
    FAISS_THROW_MSG("not implemented");
}

void IndexFlatCodes::check_compatible_for_merge(const Index& otherIndex) const {
    // minimal sanity checks
    const IndexFlatCodes* other =
            dynamic_cast<const IndexFlatCodes*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->d == d);
    FAISS_THROW_IF_NOT(other->code_size == code_size);
    FAISS_THROW_IF_NOT(other->qua_size == qua_size);
    FAISS_THROW_IF_NOT_MSG(
            typeid(*this) == typeid(*other),
            "can only merge indexes of the same type");
    
    FAISS_THROW_IF_NOT_MSG(
            include_quality == other->include_quality,
            "both index must have same include_quality status"
    );
}

void IndexFlatCodes::merge_from(Index& otherIndex, idx_t add_id) {
    FAISS_THROW_IF_NOT_MSG(add_id == 0, "cannot set ids in FlatCodes index");
    check_compatible_for_merge(otherIndex);
    IndexFlatCodes* other = static_cast<IndexFlatCodes*>(&otherIndex);
    codes.resize((ntotal + other->ntotal) * code_size);
    memcpy(codes.data() + (ntotal * code_size),
           other->codes.data(),
           other->ntotal * code_size); 

    if (get_include_quality()) {
        if (other->qualities.size() > 0) {
            qualities.resize((ntotal + other->ntotal) * qua_size); /// resize qualtity array
            memcpy(qualities.data() + (ntotal * qua_size), 
                other->qualities.data(),
                other->ntotal * qua_size);
        }
        else {
            FAISS_THROW_MSG("otherIndex not has quality array, please update it");
        }
    }
    ntotal += other->ntotal;
    other->reset();
}

CodePacker* IndexFlatCodes::get_CodePacker() const {
    FAISS_THROW_IF_NOT_MSG(include_quality == false, "get_CodePacker for IndexFlatCodes not support for quality now");
    return new CodePackerFlat(code_size);
}

void IndexFlatCodes::permute_entries(const idx_t* perm) {
    std::vector<uint8_t> new_codes(codes.size());
    std::vector<uint8_t> new_qualities(qualities.size());

    for (idx_t i = 0; i < ntotal; i++) {
        memcpy(new_codes.data() + i * code_size,
               codes.data() + perm[i] * code_size,
               code_size);
        if (get_include_quality()){
            memcpy(new_qualities.data() + i * qua_size, 
                    qualities.data() + perm[i] * qua_size,
                    qua_size);
        }

    }
    std::swap(codes, new_codes);

    if (get_include_quality()){
        std::swap(qualities, new_qualities);
    }
}

} // namespace faiss
