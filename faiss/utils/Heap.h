/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * C++ support for heaps. The set of functions is tailored for efficient
 * similarity search.
 *
 * There is no specific object for a heap, and the functions that operate on a
 * single heap are inlined, because heaps are often small. More complex
 * functions are implemented in Heaps.cpp
 *
 * All heap functions rely on a C template class that define the type of the
 * keys and values and their ordering (increasing with CMax and decreasing with
 * Cmin). The C types are defined in ordered_key_value.h
 */

#ifndef FAISS_Heap_h
#define FAISS_Heap_h

#include <climits>
#include <cmath>
#include <cstring>

#include <stdint.h>
#include <cassert>
#include <cstdio>

#include <limits>

#include <faiss/utils/ordered_key_value.h>

namespace faiss {

/*******************************************************************
 * Basic heap ops: push and pop
 *******************************************************************/

/** Pops the top element from the heap defined by bh_val[0..k-1] and
 * bh_ids[0..k-1].  on output the element at k-1 is undefined.
 */
template <class C>
inline void heap_pop(size_t k, typename C::T* bh_val, typename C::TI* bh_ids) {
    bh_val--; /* Use 1-based indexing for easier node->child translation */
    bh_ids--;
    typename C::T val = bh_val[k];
    typename C::TI id = bh_ids[k];
    size_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1; /// x2 
        i2 = i1 + 1;
        if (i1 > k)
            break;
        if ((i2 == k + 1) || // i1 = leaf
            C::cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2])) { // go less branch
            if (C::cmp2(val, bh_val[i1], id, bh_ids[i1])) { // 
                break;
            }
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            i = i1;
        } else {
            if (C::cmp2(val, bh_val[i2], id, bh_ids[i2])) {
                break;
            }
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            i = i2;
        }
    }
    bh_val[i] = bh_val[k];
    bh_ids[i] = bh_ids[k];
}

/** Pops the top element from the heap defined by bh_val[0..k-1] and
 * bh_ids[0..k-1].  on output the element at k-1 is undefined.
 */
template <class C>
inline void heap_pop_quality(size_t k, typename C::T* bh_val, typename C::TI* bh_ids, typename C::T* bh_qua) {
    bh_val--; /* Use 1-based indexing for easier node->child translation */
    bh_ids--;
    bh_qua--;
    typename C::T val = bh_val[k];
    typename C::TI id = bh_ids[k];
    typename C::T qua = bh_qua[k];

    size_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1; /// x2 
        i2 = i1 + 1;
        if (i1 > k)
            break;
        if ((i2 == k + 1) || // i1 = leaf
            C::cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2])) { // go less branch
            if (C::cmp2(val, bh_val[i1], id, bh_ids[i1])) { // 
                break;
            }
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            bh_qua[i] = bh_qua[i1];
            i = i1;
        } else {
            if (C::cmp2(val, bh_val[i2], id, bh_ids[i2])) {
                break;
            }
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            bh_qua[i] = bh_qua[i2];
            i = i2;
        }
    }
    bh_val[i] = bh_val[k];
    bh_ids[i] = bh_ids[k];
    bh_qua[i] = bh_qua[k];
}

/** Pushes the element (val, ids) into the heap bh_val[0..k-2] and
 * bh_ids[0..k-2].  on output the element at k-1 is defined.
 */
template <class C>
inline void heap_push(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T val,
        typename C::TI id) {
    bh_val--; /* Use 1-based indexing for easier node->child translation */
    bh_ids--;
    size_t i = k, i_father;
    while (i > 1) {
        i_father = i >> 1;
        if (!C::cmp2(val, bh_val[i_father], id, bh_ids[i_father])) {
            /* the heap structure is ok */
            break;
        }
        bh_val[i] = bh_val[i_father];
        bh_ids[i] = bh_ids[i_father];
        i = i_father;
    }
    bh_val[i] = val;
    bh_ids[i] = id;
}
/** Pushes the element (val, ids) into the heap bh_val[0..k-2] and
 * bh_ids[0..k-2].  on output the element at k-1 is defined.
 */
template <class C>
inline void heap_push_quality(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T* bh_qua,
        typename C::T val,
        typename C::TI id,
        typename C::T qua) {
    bh_val--; /* Use 1-based indexing for easier node->child translation */
    bh_ids--;
    bh_qua--;
    size_t i = k, i_father;
    while (i > 1) {
        i_father = i >> 1;
        if (!C::cmp2(val, bh_val[i_father], id, bh_ids[i_father])) {
            /* the heap structure is ok */
            break;
        }
        bh_val[i] = bh_val[i_father];
        bh_ids[i] = bh_ids[i_father];
        bh_qua[i] = bh_qua[i_father];
        i = i_father;
    }
    bh_val[i] = val;
    bh_ids[i] = id;
    bh_qua[i] = qua;
}

/**
 * Replaces the top element from the heap defined by bh_val[0..k-1] and
 * bh_ids[0..k-1], and for identical bh_val[] values also sorts by bh_ids[]
 * values.
 */
template <class C>
inline void heap_replace_top(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T val,
        typename C::TI id) {
    bh_val--; /* Use 1-based indexing for easier node->child translation */
    bh_ids--;
    size_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1;
        i2 = i1 + 1;
        if (i1 > k) {
            break;
        }

        // Note that C::cmp2() is a bool function answering
        // `(a1 > b1) || ((a1 == b1) && (a2 > b2))` for max
        // heap and same with the `<` sign for min heap.
        if ((i2 == k + 1) ||
            C::cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2])) {
            if (C::cmp2(val, bh_val[i1], id, bh_ids[i1])) {
                break;
            }
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            i = i1;
        } else {
            if (C::cmp2(val, bh_val[i2], id, bh_ids[i2])) {
                break;
            }
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            i = i2;
        }
    }
    bh_val[i] = val;
    bh_ids[i] = id;
}
/**
 * Replaces the top element from the heap defined by bh_val[0..k-1] and
 * bh_ids[0..k-1], and for identical bh_val[] values also sorts by bh_ids[]
 * values.
 */
template <class C>
inline void heap_replace_top_quality(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T* bh_qua,
        typename C::T val,
        typename C::TI id,
        typename C::T qua) {
    bh_val--; /* Use 1-based indexing for easier node->child translation */
    bh_ids--;
    bh_qua--;
    size_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1;
        i2 = i1 + 1;
        if (i1 > k) {
            break;
        }

        // Note that C::cmp2() is a bool function answering
        // `(a1 > b1) || ((a1 == b1) && (a2 > b2))` for max
        // heap and same with the `<` sign for min heap.
        if ((i2 == k + 1) ||
            C::cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2])) {
            if (C::cmp2(val, bh_val[i1], id, bh_ids[i1])) {
                break;
            }
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            bh_qua[i] = bh_qua[i1];
            i = i1;
        } else {
            if (C::cmp2(val, bh_val[i2], id, bh_ids[i2])) {
                break;
            }
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            bh_qua[i] = bh_qua[i2];
            i = i2;
        }
    }
    bh_val[i] = val;
    bh_ids[i] = id;
    bh_qua[i] = qua;
}

/* Partial instanciation for heaps with TI = int64_t */

template <typename T>
inline void minheap_pop(size_t k, T* bh_val, int64_t* bh_ids) {
    heap_pop<CMin<T, int64_t>>(k, bh_val, bh_ids);
}
template <typename T>
inline void minheap_pop_quality(size_t k, T* bh_val, int64_t* bh_ids, T* bh_qua) {
    heap_pop_quality<CMin<T, int64_t>>(k, bh_val, bh_ids, bh_qua);
}

template <typename T>
inline void minheap_push(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        T val,
        int64_t ids) {
    heap_push<CMin<T, int64_t>>(k, bh_val, bh_ids, val, ids);
}
template <typename T>
inline void minheap_push_quality(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        T* bh_qua,
        T val,
        int64_t ids,
        T qua) {
    heap_push_quality<CMin<T, int64_t>>(k, bh_val, bh_ids, bh_qua, val, ids, qua);
}

template <typename T>
inline void minheap_replace_top(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        T val,
        int64_t ids) {
    heap_replace_top<CMin<T, int64_t>>(k, bh_val, bh_ids, val, ids);
}

template <typename T>
inline void minheap_replace_top_quality(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        T* bh_qua,
        T val,
        int64_t ids,
        T qua) {
    heap_replace_top_quality<CMin<T, int64_t>>(k, bh_val, bh_ids, bh_qua, val, ids, qua);
}

template <typename T>
inline void maxheap_pop(size_t k, T* bh_val, int64_t* bh_ids) {
    heap_pop<CMax<T, int64_t>>(k, bh_val, bh_ids);
}

template <typename T>
inline void maxheap_pop_quality(size_t k, T* bh_val, int64_t* bh_ids, T* bh_qua) {
    heap_pop<CMax<T, int64_t>>(k, bh_val, bh_ids, bh_qua);
}

template <typename T>
inline void maxheap_push(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        T val,
        int64_t ids) {
    heap_push<CMax<T, int64_t>>(k, bh_val, bh_ids, val, ids);
}

template <typename T>
inline void maxheap_push_quality(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        T* bh_qua,
        T val,
        int64_t ids,
        T qua) {
    heap_push_quality<CMax<T, int64_t>>(k, bh_val, bh_ids, bh_qua, val, ids, qua);
}

template <typename T>
inline void maxheap_replace_top(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        T val,
        int64_t ids) {
    heap_replace_top<CMax<T, int64_t>>(k, bh_val, bh_ids, val, ids);
}

template <typename T>
inline void maxheap_replace_top_quality(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        T* bh_qua,
        T val,
        int64_t ids,
        T qua) {
    heap_replace_top_quality<CMax<T, int64_t>>(k, bh_val, bh_ids, bh_qua, val, ids, qua);
}

/*******************************************************************
 * Heap initialization
 *******************************************************************/

/* Initialization phase for the heap (with unconditionnal pushes).
 * Store k0 elements in a heap containing up to k values. Note that
 * (bh_val, bh_ids) can be the same as (x, ids) */
template <class C>
inline void heap_heapify(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        const typename C::T* x = nullptr,
        const typename C::TI* ids = nullptr,
        size_t k0 = 0) {
    if (k0 > 0)
        assert(x);

    if (ids) {
        for (size_t i = 0; i < k0; i++)
            heap_push<C>(i + 1, bh_val, bh_ids, x[i], ids[i]);
    } else {
        for (size_t i = 0; i < k0; i++)
            heap_push<C>(i + 1, bh_val, bh_ids, x[i], i);
    }

    for (size_t i = k0; i < k; i++) {
        bh_val[i] = C::neutral();
        bh_ids[i] = -1;
    }
}

/* Initialization phase for the heap (with unconditionnal pushes).
 * Store k0 elements in a heap containing up to k values. Note that
 * (bh_val, bh_ids) can be the same as (x, ids) */
template <class C>
inline void heap_heapify_quality(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T* bh_qua,
        const typename C::T* x = nullptr,
        const typename C::TI* ids = nullptr,
        const typename C::T* r_qua = nullptr,
        size_t k0 = 0) {
    if (k0 > 0)
        assert(x);

    if (ids) {
        for (size_t i = 0; i < k0; i++)
            heap_push_quality<C>(i + 1, bh_val, bh_ids, bh_qua, x[i], ids[i], r_qua[i]);
    } else {
        for (size_t i = 0; i < k0; i++)
            heap_push_quality<C>(i + 1, bh_val, bh_ids, bh_qua, x[i], i, r_qua[i]);
    }

    for (size_t i = k0; i < k; i++) {
        bh_val[i] = C::neutral();
        bh_ids[i] = -1;
        bh_qua[i] = C::neutral();
    }
}

template <typename T>
inline void minheap_heapify(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        const T* x = nullptr,
        const int64_t* ids = nullptr,
        size_t k0 = 0) {
    heap_heapify<CMin<T, int64_t>>(k, bh_val, bh_ids, x, ids, k0);
}

template <typename T>
inline void minheap_heapify_quality(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        T* bh_qua,
        const T* x = nullptr,
        const int64_t* ids = nullptr,
        const T* r_qua = nullptr,
        size_t k0 = 0) {
    heap_heapify_quality<CMin<T, int64_t>>(k, bh_val, bh_ids, bh_qua, x, ids, r_qua, k0);
}

template <typename T>
inline void maxheap_heapify(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        const T* x = nullptr,
        const int64_t* ids = nullptr,
        size_t k0 = 0) {
    heap_heapify<CMax<T, int64_t>>(k, bh_val, bh_ids, x, ids, k0);
}

template <typename T>
inline void maxheap_heapify(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        T* bh_qua,
        const T* x = nullptr,
        const int64_t* ids = nullptr,
        const T* r_qua = nullptr,
        size_t k0 = 0) {
    heap_heapify_quality<CMax<T, int64_t>>(k, bh_val, bh_ids, bh_qua, x, ids, r_qua, k0);
}

/*******************************************************************
 * Add n elements to the heap
 *******************************************************************/

/* Add some elements to the heap  */
template <class C>
inline void heap_addn(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        const typename C::T* x,
        const typename C::TI* ids,
        size_t n) {
    size_t i;
    if (ids)
        for (i = 0; i < n; i++) {
            if (C::cmp(bh_val[0], x[i])) {
                heap_replace_top<C>(k, bh_val, bh_ids, x[i], ids[i]);
            }
        }
    else
        for (i = 0; i < n; i++) {
            if (C::cmp(bh_val[0], x[i])) {
                heap_replace_top<C>(k, bh_val, bh_ids, x[i], i);
            }
        }
}

/* Add some elements to the heap  */
template <class C>
inline void heap_addn_quality(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T* bh_qua,
        const typename C::T* x,
        const typename C::TI* ids,
        const typename C::T* r_qua,
        size_t n) {
    size_t i;
    if (ids)
        for (i = 0; i < n; i++) {
            if (C::cmp(bh_val[0], x[i])) {
                heap_replace_top_quality<C>(k, bh_val, bh_ids, bh_qua, x[i], ids[i], r_qua[i]);
            }
        }
    else
        for (i = 0; i < n; i++) {
            if (C::cmp(bh_val[0], x[i])) {
                heap_replace_top_quality<C>(k, bh_val, bh_ids, bh_qua, x[i], i, r_qua[i]);
            }
        }
}

/* Partial instanciation for heaps with TI = int64_t */

template <typename T>
inline void minheap_addn(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        const T* x,
        const int64_t* ids,
        size_t n) {
    heap_addn<CMin<T, int64_t>>(k, bh_val, bh_ids, x, ids, n);
}

template <typename T>
inline void minheap_addn_quality(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        T* bh_qua,
        const T* x,
        const int64_t* ids,
        const T* r_qua,
        size_t n) {
    heap_addn_quality<CMin<T, int64_t>>(k, bh_val, bh_ids, bh_qua, x, ids, r_qua, n);
}

template <typename T>
inline void maxheap_addn(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        const T* x,
        const int64_t* ids,
        size_t n) {
    heap_addn<CMax<T, int64_t>>(k, bh_val, bh_ids, x, ids, n);
}

template <typename T>
inline void maxheap_addn_quality(
        size_t k,
        T* bh_val,
        int64_t* bh_ids,
        T* bh_qua,
        const T* x,
        const int64_t* ids,
        const T* r_qua,
        size_t n) {
    heap_addn_quality<CMax<T, int64_t>>(k, bh_val, bh_ids, bh_qua, x, ids, r_qua, n);
}

/*******************************************************************
 * Heap finalization (reorder elements)
 *******************************************************************/

/* This function maps a binary heap into a sorted structure.
   It returns the number  */
template <typename C>
inline size_t heap_reorder(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids) {
    size_t i, ii;

    for (i = 0, ii = 0; i < k; i++) {
        /* top element should be put at the end of the list */
        typename C::T val = bh_val[0];
        typename C::TI id = bh_ids[0];

        /* boundary case: we will over-ride this value if not a true element */
        heap_pop<C>(k - i, bh_val, bh_ids);
        bh_val[k - ii - 1] = val;
        bh_ids[k - ii - 1] = id;
        if (id != -1)
            ii++;
    }
    /* Count the number of elements which are effectively returned */
    size_t nel = ii;

    memmove(bh_val, bh_val + k - ii, ii * sizeof(*bh_val));
    memmove(bh_ids, bh_ids + k - ii, ii * sizeof(*bh_ids));

    for (; ii < k; ii++) {
        bh_val[ii] = C::neutral();
        bh_ids[ii] = -1;
    }
    return nel;
}
/* This function maps a binary heap into a sorted structure.
   It returns the number  */
template <typename C>
inline size_t heap_reorder_quality(
        size_t k,
        typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T* bh_qua) {
    size_t i, ii;

    for (i = 0, ii = 0; i < k; i++) {
        /* top element should be put at the end of the list */
        typename C::T val = bh_val[0];
        typename C::TI id = bh_ids[0];
        typename C::T qua = bh_qua[0];

        /* boundary case: we will over-ride this value if not a true element */
        heap_pop_quality<C>(k - i, bh_val, bh_ids, bh_qua);
        bh_val[k - ii - 1] = val;
        bh_ids[k - ii - 1] = id;
        bh_qua[k - ii - 1] = qua;
        if (id != -1)
            ii++;
    }
    /* Count the number of elements which are effectively returned */
    size_t nel = ii;

    memmove(bh_val, bh_val + k - ii, ii * sizeof(*bh_val));
    memmove(bh_ids, bh_ids + k - ii, ii * sizeof(*bh_ids));
    memmove(bh_qua, bh_qua + k - ii, ii * sizeof(*bh_qua));

    for (; ii < k; ii++) {
        bh_val[ii] = C::neutral();
        bh_ids[ii] = -1;
        bh_qua[ii] = C::neutral();
    }
    return nel;
}

template <typename T>
inline size_t minheap_reorder(size_t k, T* bh_val, int64_t* bh_ids) {
    return heap_reorder<CMin<T, int64_t>>(k, bh_val, bh_ids);
}

template <typename T>
inline size_t minheap_reorder_quality(size_t k, T* bh_val, int64_t* bh_ids, T* bh_qua) {
    return heap_reorder_quality<CMin<T, int64_t>>(k, bh_val, bh_ids, bh_qua);
}

template <typename T>
inline size_t maxheap_reorder(size_t k, T* bh_val, int64_t* bh_ids) {
    return heap_reorder<CMax<T, int64_t>>(k, bh_val, bh_ids);
}

template <typename T>
inline size_t maxheap_reorder_quality(size_t k, T* bh_val, int64_t* bh_ids, T* bh_qua) {
    return heap_reorder_quality<CMax<T, int64_t>>(k, bh_val, bh_ids, bh_qua);
}

/*******************************************************************
 * Operations on heap arrays
 *******************************************************************/

/** a template structure for a set of [min|max]-heaps it is tailored
 * so that the actual data of the heaps can just live in compact
 * arrays.
 */
template <typename C>
struct HeapArray {
    typedef typename C::TI TI;
    typedef typename C::T T;

    size_t nh; ///< number of heaps
    size_t k;  ///< allocated size per heap
    TI* ids;   ///< identifiers (size nh * k)
    T* val;    ///< values (distances or similarities), size nh * k

    /// Return the list of values for a heap
    T* get_val(size_t key) {
        return val + key * k;
    }

    /// Correspponding identifiers
    TI* get_ids(size_t key) {
        return ids + key * k;
    }

    /// prepare all the heaps before adding
    void heapify();

    /** add nj elements to heaps i0:i0+ni, with sequential ids
     *
     * @param nj    nb of elements to add to each heap
     * @param vin   elements to add, size ni * nj
     * @param j0    add this to the ids that are added
     * @param i0    first heap to update
     * @param ni    nb of elements to update (-1 = use nh)
     */
    void addn(
            size_t nj,
            const T* vin,
            TI j0 = 0,
            size_t i0 = 0,
            int64_t ni = -1);

    /** same as addn
     *
     * @param id_in     ids of the elements to add, size ni * nj
     * @param id_stride stride for id_in
     */
    void addn_with_ids(
            size_t nj,
            const T* vin,
            const TI* id_in = nullptr,
            int64_t id_stride = 0,
            size_t i0 = 0,
            int64_t ni = -1);

    /** same as addn_with_ids, but for just a subset of queries
     *
     * @param nsubset  number of query entries to update
     * @param subset   indexes of queries to update, in 0..nh-1, size nsubset
     */
    void addn_query_subset_with_ids(
            size_t nsubset,
            const TI* subset,
            size_t nj,
            const T* vin,
            const TI* id_in = nullptr,
            int64_t id_stride = 0);

    /// reorder all the heaps
    void reorder();

    /** this is not really a heap function. It just finds the per-line
     *   extrema of each line of array D
     * @param vals_out    extreme value of each line (size nh, or NULL)
     * @param idx_out     index of extreme value (size nh or NULL)
     */
    void per_line_extrema(T* vals_out, TI* idx_out) const;
};

/** a template structure for a set of [min|max]-heaps it is tailored
 * so that the actual data of the heaps can just live in compact
 * arrays.
 */
template <typename C>
struct HeapArrayQuality {
    typedef typename C::TI TI;
    typedef typename C::T T;

    size_t nh; ///< number of heaps
    size_t k;  ///< allocated size per heap
    TI* ids;   ///< identifiers (size nh * k)
    T* val;    ///< values (distances or similarities), size nh * k
    T* qua;    ///< value quality, size nh*k

    /// Return the list of values for a heap
    T* get_val(size_t key) {
        return val + key * k;
    }

    /// Correspponding identifiers
    TI* get_ids(size_t key) {
        return ids + key * k;
    }

    T* get_qua(size_t key) {
        return qua + key * k; 
    }

    /// prepare all the heaps before adding
    void heapify();

    /** add nj elements to heaps i0:i0+ni, with sequential ids
     *
     * @param nj    nb of elements to add to each heap
     * @param vin   elements to add, size ni * nj
     * @param vin   elements quality to add, size ni * nj
     * @param j0    add this to the ids that are added
     * @param i0    first heap to update
     * @param ni    nb of elements to update (-1 = use nh)
     */
    void addn(
            size_t nj,
            const T* vin,
            const T* qin,
            TI j0 = 0,
            size_t i0 = 0,
            int64_t ni = -1);

    /** same as addn
     *
     * @param id_in     ids of the elements to add, size ni * nj
     * @param id_stride stride for id_in
     */
    void addn_with_ids(
            size_t nj,
            const T* vin,
            const T* qin,
            const TI* id_in = nullptr,
            int64_t id_stride = 0,
            size_t i0 = 0,
            int64_t ni = -1);

    /** same as addn_with_ids, but for just a subset of queries
     *
     * @param nsubset  number of query entries to update
     * @param subset   indexes of queries to update, in 0..nh-1, size nsubset
     */
    void addn_query_subset_with_ids(
            size_t nsubset,
            const TI* subset,
            size_t nj,
            const T* vin,
            const T* qin,
            const TI* id_in = nullptr,
            int64_t id_stride = 0);

    /// reorder all the heaps
    void reorder();

    /** this is not really a heap function. It just finds the per-line
     *   extrema of each line of array D
     * @param vals_out    extreme value of each line (size nh, or NULL)
     * @param idx_out     index of extreme value (size nh or NULL)
     */
    void per_line_extrema(T* vals_out, T* quas_out, TI* idx_out) const;
};

/* Define useful heaps */
typedef HeapArray<CMin<float, int64_t>> float_minheap_array_t;
typedef HeapArray<CMin<int, int64_t>> int_minheap_array_t;

typedef HeapArray<CMax<float, int64_t>> float_maxheap_array_t;
typedef HeapArray<CMax<int, int64_t>> int_maxheap_array_t;

/* Define useful heaps */
typedef HeapArrayQuality<CMin<float, int64_t>> float_minheap_quality_array_t;
typedef HeapArrayQuality<CMin<int, int64_t>> int_minheap_quality_array_t;

typedef HeapArrayQuality<CMax<float, int64_t>> float_maxheap_quality_array_t;
typedef HeapArrayQuality<CMax<int, int64_t>> int_maxheap_quality_array_t;

// The heap templates are instantiated explicitly in Heap.cpp

/*********************************************************************
 * Indirect heaps: instead of having
 *
 *          node i = (bh_ids[i], bh_val[i]),
 *
 * in indirect heaps,
 *
 *          node i = (bh_ids[i], bh_val[bh_ids[i]]),
 *
 *********************************************************************/

template <class C>
inline void indirect_heap_pop(
        size_t k,
        const typename C::T* bh_val,
        typename C::TI* bh_ids) {
    bh_ids--; /* Use 1-based indexing for easier node->child translation */
    typename C::T val = bh_val[bh_ids[k]];
    size_t i = 1;
    while (1) {
        size_t i1 = i << 1;
        size_t i2 = i1 + 1;
        if (i1 > k)
            break;
        typename C::TI id1 = bh_ids[i1], id2 = bh_ids[i2];
        if (i2 == k + 1 || C::cmp(bh_val[id1], bh_val[id2])) {
            if (C::cmp(val, bh_val[id1]))
                break;
            bh_ids[i] = id1;
            i = i1;
        } else {
            if (C::cmp(val, bh_val[id2]))
                break;
            bh_ids[i] = id2;
            i = i2;
        }
    }
    bh_ids[i] = bh_ids[k];
}

template <class C>
inline void indirect_heap_pop_quality(
        size_t k,
        const typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T* bh_qua) {
    bh_ids--; /* Use 1-based indexing for easier node->child translation */
    typename C::T val = bh_val[bh_ids[k]];
    typename C::T qua = bh_qua[bh_ids[k]];
    size_t i = 1;
    while (1) {
        size_t i1 = i << 1;
        size_t i2 = i1 + 1;
        if (i1 > k)
            break;
        typename C::TI id1 = bh_ids[i1], id2 = bh_ids[i2];
        if (i2 == k + 1 || C::cmp(bh_val[id1], bh_val[id2])) {
            if (C::cmp(val, bh_val[id1]))
                break;
            bh_ids[i] = id1;
            i = i1;
        } else {
            if (C::cmp(val, bh_val[id2]))
                break;
            bh_ids[i] = id2;
            i = i2;
        }
    }
    bh_ids[i] = bh_ids[k];
}

template <class C>
inline void indirect_heap_push(
        size_t k,
        const typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::TI id) {
    bh_ids--; /* Use 1-based indexing for easier node->child translation */
    typename C::T val = bh_val[id];
    size_t i = k;
    while (i > 1) {
        size_t i_father = i >> 1;
        if (!C::cmp(val, bh_val[bh_ids[i_father]]))
            break;
        bh_ids[i] = bh_ids[i_father];
        i = i_father;
    }
    bh_ids[i] = id;
}
template <class C>
inline void indirect_heap_push_quality(
        size_t k,
        const typename C::T* bh_val,
        typename C::TI* bh_ids,
        typename C::T* bh_qua,
        typename C::TI id) {
    bh_ids--; /* Use 1-based indexing for easier node->child translation */
    typename C::T val = bh_val[id];
    size_t i = k;
    while (i > 1) {
        size_t i_father = i >> 1;
        if (!C::cmp(val, bh_val[bh_ids[i_father]]))
            break;
        bh_ids[i] = bh_ids[i_father];
        i = i_father;
    }
    bh_ids[i] = id;
}

/** Merge result tables from several shards. The per-shard results are assumed
 * to be sorted. Note that the C comparator is reversed w.r.t. the usual top-k
 * element heap because we want the best (ie. lowest for L2) result to be on
 * top, not the worst. Also, it needs to hold an index of a shard id (ie.
 * usually int32 is more than enough).
 *
 * @param all_distances  size (nshard, n, k)
 * @param all_labels     size (nshard, n, k)
 * @param distances      output distances, size (n, k)
 * @param labels         output labels, size (n, k)
 */
template <class idx_t, class C>
void merge_knn_results(
        size_t n,
        size_t k,
        typename C::TI nshard,
        const typename C::T* all_distances,
        const idx_t* all_labels,
        typename C::T* distances,
        idx_t* labels);

/** Merge result tables from several shards. The per-shard results are assumed
 * to be sorted. Note that the C comparator is reversed w.r.t. the usual top-k
 * element heap because we want the best (ie. lowest for L2) result to be on
 * top, not the worst. Also, it needs to hold an index of a shard id (ie.
 * usually int32 is more than enough).
 *
 * @param all_distances  size (nshard, n, k)
 * @param all_labels     size (nshard, n, k)
 * @param all_qualities     size (nshard, n, k)
 * @param distances      output distances, size (n, k)
 * @param qualities      output qualities, size (n, k)
 * @param labels         output labels, size (n, k)
 */
template <class idx_t, class C>
void merge_knn_results_quality(
        size_t n,
        size_t k,
        typename C::TI nshard,
        const typename C::T* all_distances,
        const typename C::T* all_qualities,
        const idx_t* all_labels,
        typename C::T* distances,
        typename C::T* qualities,
        idx_t* labels);

} // namespace faiss

#endif /* FAISS_Heap_h */
