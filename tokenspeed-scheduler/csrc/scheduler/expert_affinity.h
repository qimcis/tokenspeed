// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace tokenspeed {

// A bounded, admission-independent selection step. Missing signatures are FIFO
// barriers. The caller adds only successfully admitted routes to the union.
struct ExpertAffinityCandidate {
    std::span<const std::uint64_t> keys;
    std::int32_t deferrals;
};

// Immutable logical-key to expected per-rank work, published as one epoch.
struct ExpertPlacementModel {
    std::uint64_t epoch;
    std::size_t num_ranks;
    std::unordered_map<std::uint64_t, std::vector<double>> rank_weights;
    bool Covers(std::span<const std::uint64_t> keys) const {
        return std::ranges::all_of(keys, [this](auto key) { return rank_weights.contains(key); });
    }
    void Add(std::span<const std::uint64_t> keys, std::vector<double>& loads) const {
        for (auto key : keys) {
            const auto& weights = rank_weights.at(key);
            for (std::size_t r = 0; r < num_ranks; ++r) loads[r] += weights[r];
        }
    }
};

inline std::size_t SelectExpertAffinity(std::span<const ExpertAffinityCandidate> candidates,
                                        const std::unordered_set<std::uint64_t>& expert_union,
                                        std::int32_t max_deferrals, const ExpertPlacementModel* placement = nullptr,
                                        std::span<const double> rank_loads = {}) {
    std::size_t selected = 0;
    std::int32_t oldest = max_deferrals - 1;
    double best_cost = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        if (candidate.keys.empty()) break;
        if (candidate.deferrals >= max_deferrals) {
            if (candidate.deferrals > oldest) {
                oldest = candidate.deferrals;
                selected = i;
            }
            continue;
        }
        if (oldest >= max_deferrals || expert_union.empty()) continue;
        double cost = 0;
        if (placement) {
            std::vector<double> loads(rank_loads.begin(), rank_loads.end());
            placement->Add(candidate.keys, loads);
            cost = *std::max_element(loads.begin(), loads.end());
        } else {
            for (const auto key : candidate.keys) cost += !expert_union.contains(key);
        }
        if (cost < best_cost) {
            best_cost = cost;
            selected = i;
        }
    }
    return selected;
}

}  // namespace tokenspeed
