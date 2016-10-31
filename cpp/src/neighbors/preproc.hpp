//
//  nn_index.hpp
//  Dig
//
//  Created by DB on 2016-9-15
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef __PREPROC_HPP
#define __PREPROC_HPP

#include <memory>
#include "assert.h"

#include "Dense"

#include "cluster.hpp"
#include "nn_search.hpp"
#include "nn_utils.hpp"
#include "array_utils.hpp"
#include "eigen_utils.hpp"
#include "flat_store.hpp"

// #include "debug_utils.hpp"

namespace nn {


// ================================================================
// Preprocessing
// ================================================================

struct IdentityPreproc {
    static constexpr int align_bytes = 0;

    // ------------------------ ctors
    template<class RowMatrixT> IdentityPreproc(const RowMatrixT& X) {}
    IdentityPreproc() = default;

    // ------------------------ identity funcs
    template<class VectorT>
    void preprocess(const VectorT& query, typename VectorT::Scalar* out) const {}
    template<class VectorT>
    VectorT& preprocess(VectorT& query) const { return query; }

    template<class RowMatrixT, class RowMatrixT2>
    void preprocess_batch(const RowMatrixT& queries, RowMatrixT2& out) const {}
    template<class RowMatrixT>
    RowMatrixT& preprocess_batch(RowMatrixT& queries) const { return queries; }

    template<class RowMatrixT>
    RowMatrixT& preprocess_data(RowMatrixT& X) const { return X; }
};

template<class ScalarT, int AlignBytes=kDefaultAlignBytes>
struct ReorderPreproc { // TODO could avoid passing ScalarT template arg
    // ------------------------ consts
    using Scalar = ScalarT;
    using Index = int32_t;
    static constexpr int max_pad_elements = AlignBytes / sizeof(Scalar);
    static constexpr int align_bytes = AlignBytes;

    // ------------------------ ctors
    ReorderPreproc() = default;

    template<class RowMatrixT> ReorderPreproc(const RowMatrixT& X):
            _order(nn::order_col_variance(X))
    {
        assert(_length() > max_pad_elements);
    }

    // ------------------------ one query
    // assumes that out already has appropriate padding at the end
    template<class VectorT> void preprocess(const VectorT& query,
        typename VectorT::Scalar* out) const
    {
        assert(_length() == _aligned_length(query.size()));
        reorder_query(query, _order, out);
    }
    template<class VectorT> auto preprocess(const VectorT& query) const
        -> typename mat_traits<VectorT>::VectorT
    {
        typename mat_traits<VectorT>::VectorT out(_length());
        // out.segment(query.size(), _length() - query.size()) = 0;
        out.setZero();
        preprocess(query, out.data());
        return out;
    }

    // ------------------------ batch of queries
    template<class RowMatrixT>
    void preprocess_batch(const RowMatrixT& queries, RowMatrixT& out) const {
        assert(queries.IsRowMajor);
        assert(out.IsRowMajor);
        assert(queries.rows() <= out.rows());
        assert(queries.cols() <= out.cols());
        for (Index i = 0; i < queries.rows(); i++) {
            preprocess(queries.row(i), out.row(i).data());
        }
    }
    template<class RowMatrixT>
    RowMatrixT preprocess_batch(const RowMatrixT& queries) const {
        assert(false); // verify we're not actually calling this yet
        RowMatrixT out(queries.rows(), _length());
        // out.topRightCorner(queries.rows(), max_pad_elements).setZero(); // TODO uncomment
        out.setZero();
        preprocess_batch(queries, out);
        return out;
    }

    // ------------------------ data mat
    template<class RowMatrixT> RowMatrixT preprocess_data(
        const RowMatrixT& X) const
    {
        assert(X.IsRowMajor);
        assert(X.cols() <= _length());
        assert(X.cols() > 0);

        // create mat to return and ensure final cols are zeroed
        RowMatrixT out(X.rows(), _length());
        out.topRightCorner(X.rows(), max_pad_elements).setZero();

        // preprocess it the same way as queries
        preprocess_batch(X, out);
        return out;
    }
private:
    Index _aligned_length(int64_t n) const {
        return static_cast<Index>(aligned_length<Scalar, align_bytes>(n));
    }
    Index _length() const { return _aligned_length(_order.size()); }
    std::vector<Index> _order;
};

} // namespace nn

#endif // __PREPROC_HPP
