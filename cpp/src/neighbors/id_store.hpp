//
//  nn_index.hpp
//  Dig
//
//  Created by DB on 2016-11-5
//  Copyright (c) 2016 DB. All rights reserved.
//

#ifndef __ID_STORE_HPP
#define __ID_STORE_HPP


#include <experimental/optional>
#include <vector>

#include "macros.hpp"
#include "neighbors.hpp"


namespace nn {

// ================================================================
// FlatIdStore / Neighbor postprocessing
// ================================================================

struct IdentityIdStore {
    template<class... Args> IdentityIdStore(Args&&... args) {};
    template<class T> T postprocess(const T& neighbors) { return neighbors; }
};

template<class IdT=idx_t>
class FlatIdStore {
public:
    typedef IdT ID;

    FlatIdStore(const std::vector<ID> ids): _ids(ids) {}
    FlatIdStore(size_t len):
        // _ids(ar::range(static_cast<ID>(0), static_cast<ID>(len)))
        FlatIdStore(ar::range(static_cast<ID>(0), static_cast<ID>(len))) {}
    FlatIdStore(const optional<std::vector<ID> > ids, size_t len):
        FlatIdStore(len)
    {
        if (ids) {
            _ids = *ids;
        }
        // else { PRINT("not using ids cuz they were null"); }
    }

    FlatIdStore() = default;
    FlatIdStore(const FlatIdStore& rhs) = delete;
    // FlatIdStore(FlatIdStore&& rhs): _ids(std::move(rhs._ids)) {
    //     PRINT("move ctor")
    //     PRINT_VAR(_ids.size());
    // }

    idx_t rows() const { return static_cast<idx_t>(_ids.size()); }

    // TODO keep this protected and add it as optional arg to ctor
    std::vector<ID>& ids() { return _ids; }

    // ------------------------ neighbor postprocessing
    // ie, convert indices within storage to point IDs

    Neighbor postprocess(Neighbor n) {
        // PRINT_VAR(_ids.size());
        // DEBUGF("%lld (%g) -> %lld", n.idx, n.dist, _ids[n.idx]);
        // PRINT_VAR(n.dist);

        return Neighbor{_ids[n.idx], n.dist};
    }
    std::vector<Neighbor> postprocess(std::vector<Neighbor> neighbors) {
        auto idxs = ar::map([](auto& n) { return n.idx; }, neighbors);
        auto dists = ar::map([](auto& n) { return n.dist; }, neighbors);
        return ar::map([this](const Neighbor n) {
            return this->postprocess(n);
        }, neighbors);
    }
    std::vector<std::vector<Neighbor> > postprocess(
        std::vector<std::vector<Neighbor> > nested_neighbors) {
        return ar::map([this](const std::vector<Neighbor> neighbors) {
            return this->postprocess(neighbors);
        }, nested_neighbors);
    }

protected:
    std::vector<ID> _ids;
};

} // namespace nn
#endif // __ID_STORE_HPP
