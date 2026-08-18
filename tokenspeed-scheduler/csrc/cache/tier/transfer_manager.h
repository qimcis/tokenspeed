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

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cache/coordinator/kv_cache_coordinator.h"
#include "cache/tier/transfer.h"

namespace tokenspeed {

// Owns the mechanics and asynchronous lifetime of transfers between Device and
// Host cache tiers. Scheduling policy and request state transitions stay in
// Scheduler.
class TierTransferManager {
public:
    struct FailedStoreLoad {
        std::string request_id;
        std::vector<std::pair<std::uint32_t, CacheBlockLocation>> destinations;
    };

    explicit TierTransferManager(KvCacheCoordinator& coordinator) : coordinator_{coordinator} {}

    void SetEnableL3(bool enable) { enable_l3_storage_ = enable; }

    std::optional<WriteBackOperation> StartPendingStores();
    LoadBackOperation StartPrefixLoad(std::vector<BlockTransfer> block_transfers);
    StoreLoadOperation StartStoreLoad(std::string request_id,
                                      std::vector<KvCacheCoordinator::StoreTransfer> store_transfers);

    void CompleteWriteBack(std::uint32_t op_id);
    void CompleteLoadBack(std::uint32_t op_id);
    void CompleteStoreLoad(std::uint32_t op_id);
    std::optional<FailedStoreLoad> FailStoreLoad(std::uint32_t op_id);

    bool HasStoresInFlight() const { return !write_backs_.empty(); }
    bool HasLoadBacksInFlight() const { return !load_backs_.empty() || !store_loads_.empty(); }
    bool HasAnyInFlight() const { return !write_backs_.empty() || !load_backs_.empty() || !store_loads_.empty(); }
    std::vector<std::pair<std::uint32_t, CacheBlockLocation>> DeviceLocationsReleasedOnStoreAck() const;

private:
    struct StoreTicket {
        CacheKey key;
        CacheBlockRef device_block_ref;
        CacheBlockRef host_block_ref;
    };

    struct StoreLoadTicket {
        std::string request_id;
        std::vector<KvCacheCoordinator::StoreTransfer> transfers;
    };

    std::uint32_t nextOpId() { return next_op_id_++; }
    LoadBackOperation startLoadBack(std::vector<BlockTransfer> block_transfers);
    std::vector<CacheTransfer> resolveTransfers(std::span<const BlockTransfer> block_transfers) const;

    KvCacheCoordinator& coordinator_;
    bool enable_l3_storage_{false};
    std::unordered_map<std::uint32_t, std::vector<StoreTicket>> write_backs_;
    std::unordered_set<CacheKey, CacheKeyHash> store_keys_;
    // Each transfer pins both tiers until the runtime acknowledges the copy.
    std::unordered_map<std::uint32_t, std::vector<BlockTransfer>> load_backs_;
    std::unordered_map<std::uint32_t, StoreLoadTicket> store_loads_;
    std::uint32_t next_op_id_{0};
};

}  // namespace tokenspeed
