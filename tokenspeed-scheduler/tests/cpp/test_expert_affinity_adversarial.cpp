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

#include "integration_test_helper.h"
#include "scheduler/expert_affinity.h"

#include <map>

namespace tokenspeed::test {

class ExpertAffinityAdversarialSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.max_batch_size = 6;
        cfg.max_decode_batch_size = 2;
        cfg.device_allocator.total_pages = 1024;
        cfg.cache_groups.front().total_pages = 1024;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        SetTestSnapshotPool(cfg);
        return cfg;
    }

    void Prefill(const std::vector<std::string>& ids) {
        std::vector<RequestSpec> specs;
        for (const auto& id : ids) {
            specs.push_back({.request_id = id, .tokens = {1, 2}, .max_new_tokens = 128});
        }
        Submit(specs);
        auto plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        ASSERT_EQ(FindForwardBatch(plan)->request_ids, ids);
        Complete(plan);
    }

    void Complete(const ExecutionPlan& plan) {
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        for (const auto& id : FindForwardBatch(plan)->request_ids) SendForwardDone(id, {10});
    }

    void AssertFairRotation() {
        Prefill({"a", "b", "c", "d", "e"});
        const std::vector<std::vector<std::string>> expected{
            {"a", "b"}, {"c", "d"}, {"e", "a"}, {"b", "c"}, {"d", "e"}};
        for (int cycle = 0; cycle < 3; ++cycle) {
            for (const auto& ids : expected) {
                auto plan = PlanOnce();
                ASSERT_NE(FindForwardBatch(plan), nullptr);
                EXPECT_EQ(FindForwardBatch(plan)->request_ids, ids);
                Complete(plan);
            }
        }
    }
};

TEST_F(ExpertAffinityAdversarialSuite, CappedFifoRotatesEveryResidentIncludingFirstDecodes) {
    AssertFairRotation();
}

TEST_F(ExpertAffinityAdversarialSuite, MissingHintsHaveIdenticalBoundedFifoRotation) {
    scheduler_->ConfigureExpertAffinity(6, 2, 8);
    AssertFairRotation();
}

class ExpertAffinityArrivalSuite : public ExpertAffinityAdversarialSuite, public ::testing::WithParamInterface<int> {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = ExpertAffinityAdversarialSuite::MakeConfig();
        cfg.max_decode_batch_size = GetParam();
        return cfg;
    }
};

TEST_P(ExpertAffinityArrivalSuite, ContinuousPrefillDoneArrivalsRespectCapAndResidentProgress) {
    scheduler_->ConfigureExpertAffinity(6, 2, 1);
    Prefill({"a", "b", "c", "d"});
    const auto cap = static_cast<std::size_t>(GetParam());
    for (int i = 0; i < 4; ++i) {
        auto plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        ASSERT_LE(FindForwardBatch(plan)->request_ids.size(), cap);
        Complete(plan);
    }
    const auto hint = [&](const std::string& id, std::uint64_t key) {
        return scheduler_->UpdateExpertRoute(id, scheduler_->RequestGeneration(id),
                                             scheduler_->RequestTokenSize(id) - 1, {key});
    };
    ASSERT_TRUE(hint("a", 1));
    ASSERT_TRUE(hint("b", 99));
    ASSERT_TRUE(hint("c", 1));
    std::map<std::string, int> serviced;
    for (int arrival = 0; arrival < 20; ++arrival) {
        const auto newcomer = "new-" + std::to_string(arrival);
        Prefill({newcomer});
        auto plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        const auto ids = FindForwardBatch(plan)->request_ids;
        ASSERT_LE(ids.size(), cap);
        for (const auto& id : ids) ++serviced[id];
        Complete(plan);
        for (const auto& id : ids) {
            if (id == "a" || id == "c") ASSERT_TRUE(hint(id, 1));
        }
        // Cancel each short-lived arrival even when it has not decoded. There
        // is always room for another arrival, so finite slot count cannot hide
        // starvation caused by putting every newcomer ahead of old decoders.
        SendAbortEvent(newcomer);
    }
    for (const auto& id : {"a", "b", "c", "d"}) EXPECT_GE(serviced[id], 3);
}

INSTANTIATE_TEST_SUITE_P(DecodeCaps, ExpertAffinityArrivalSuite, ::testing::Values(1, 2));

TEST_F(ExpertAffinityAdversarialSuite, CancellationAndIdentityReuseDoNotRetainServicePriority) {
    Prefill({"a", "b", "c", "d", "e"});
    auto first = PlanOnce();
    ASSERT_NE(FindForwardBatch(first), nullptr);
    ASSERT_EQ(FindForwardBatch(first)->request_ids, (std::vector<std::string>{"a", "b"}));
    Complete(first);
    SendAbortEvent("a");
    // Finished IDs are released at the next planning boundary.
    auto next = PlanOnce();
    ASSERT_NE(FindForwardBatch(next), nullptr);
    EXPECT_EQ(FindForwardBatch(next)->request_ids, (std::vector<std::string>{"c", "d"}));
    Complete(next);
    Prefill({"a"});
    auto reused = PlanOnce();
    ASSERT_NE(FindForwardBatch(reused), nullptr);
    EXPECT_EQ(FindForwardBatch(reused)->request_ids, (std::vector<std::string>{"e", "b"}));
    Complete(reused);
    auto waiting = PlanOnce();
    ASSERT_NE(FindForwardBatch(waiting), nullptr);
    EXPECT_EQ(FindForwardBatch(waiting)->request_ids, (std::vector<std::string>{"c", "d"}));
    Complete(waiting);
    auto served = PlanOnce();
    ASSERT_NE(FindForwardBatch(served), nullptr);
    EXPECT_EQ(FindForwardBatch(served)->request_ids, (std::vector<std::string>{"a", "e"}));
}

class ExpertAffinityUncappedSuite : public ExpertAffinityAdversarialSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = ExpertAffinityAdversarialSuite::MakeConfig();
        cfg.max_decode_batch_size = 0;
        return cfg;
    }
};

TEST_F(ExpertAffinityUncappedSuite, DefaultSchedulesAllResidentsInOriginalOrder) {
    const std::vector<std::string> ids{"a", "b", "c", "d", "e"};
    Prefill(ids);
    for (int round = 0; round < 8; ++round) {
        auto plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        EXPECT_EQ(FindForwardBatch(plan)->request_ids, ids);
        Complete(plan);
    }
}

TEST_F(ExpertAffinityAdversarialSuite, MixedFreshUnknownAndExpiringHintsKeepEveryResidentProgressing) {
    scheduler_->ConfigureExpertAffinity(6, 1, 1);
    Prefill({"a", "b", "c", "d", "e"});
    for (int bootstrap = 0; bootstrap < 3; ++bootstrap) {
        auto plan = PlanOnce();
        Complete(plan);
    }
    const auto hint = [&](const std::string& id, std::uint64_t key) {
        return scheduler_->UpdateExpertRoute(id, scheduler_->RequestGeneration(id),
                                             scheduler_->RequestTokenSize(id) - 1, {key});
    };
    // b's hint expires as it progresses, d is always unknown, and a/c/e
    // keep conflicting fresh supports. No hint may strand either FIFO barrier.
    ASSERT_TRUE(hint("b", 99));
    for (const auto& id : {"a", "c", "e"}) ASSERT_TRUE(hint(id, id[0] == 'c' ? 2 : 1));
    std::map<std::string, int> last_service;
    for (const auto& id : {"a", "b", "c", "d", "e"}) last_service[id] = -1;
    for (int round = 0; round < 30; ++round) {
        auto plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        const auto ids = FindForwardBatch(plan)->request_ids;
        ASSERT_EQ(ids.size(), 2u);
        Complete(plan);
        for (const auto& id : ids) {
            last_service[id] = round;
            if (id == "a" || id == "c" || id == "e") ASSERT_TRUE(hint(id, id == "c" ? 2 : 1));
        }
        // The threshold promotes affinity candidates; it cannot remove the
        // rounds needed to drain a capped batch of multiple ready requests.
        for (const auto& [id, last] : last_service) EXPECT_LE(round - last, 4) << id;
    }
}

class ExpertAffinityDecodeWorkerSuite : public ExpertAffinityAdversarialSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = ExpertAffinityAdversarialSuite::MakeConfig();
        cfg.role = Role::kD;
        cfg.cache_groups.front().transfer_policy = CacheTransferPolicy::FullSuffix;
        return cfg;
    }

    void RemoteDone(const std::string& id) {
        ExecutionEvent event;
        event.With(pd::RemotePrefillDoneEvent{id, 42});
        scheduler_->Advance(std::move(event));
    }
};

TEST_F(ExpertAffinityDecodeWorkerSuite, PendingRemotePrefillNeverEntersDecodeOrRouteSelection) {
    scheduler_->ConfigureExpertAffinity(6, 1, 8);
    for (const auto& id : {"a", "b", "c"}) {
        Submit(RequestSpec{.request_id = id, .tokens = {1, 2}, .max_new_tokens = 128});
        ExecutionEvent event;
        event.With(pd::BootstrappedEvent{id});
        scheduler_->Advance(std::move(event));
    }
    for (const auto& id : {"a", "b", "c"}) {
        auto admission = PlanOnce();
        ASSERT_NE(FindRemoteAdmission(admission), nullptr);
        EXPECT_EQ(FindRemoteAdmission(admission)->request_ids, (std::vector<std::string>{id}));
        const auto* local = FindForwardBatch(admission);
        EXPECT_TRUE(local == nullptr || local->request_ids.empty());
    }
    RemoteDone("a");
    RemoteDone("c");
    auto first = PlanOnce();
    ASSERT_NE(FindForwardBatch(first), nullptr);
    EXPECT_EQ(FindForwardBatch(first)->request_ids, (std::vector<std::string>{"a", "c"}));
    EXPECT_EQ(FindForwardBatch(first)->decode_input_ids, (std::vector<std::int32_t>{42, 42}));
    Complete(first);
    for (const auto& id : {"a", "c"}) {
        EXPECT_TRUE(scheduler_->UpdateExpertRoute(id, scheduler_->RequestGeneration(id),
                                                  scheduler_->RequestTokenSize(id) - 1, {1}));
    }
    EXPECT_FALSE(scheduler_->UpdateExpertRoute("b", scheduler_->RequestGeneration("b"), 2, {1}));
    for (int round = 0; round < 3; ++round) {
        auto waiting = PlanOnce();
        ASSERT_NE(FindForwardBatch(waiting), nullptr);
        EXPECT_EQ(FindForwardBatch(waiting)->request_ids, (std::vector<std::string>{"a", "c"}));
        Complete(waiting);
    }
    RemoteDone("b");
    bool served = false;
    for (int round = 0; round < 3; ++round) {
        auto ready = PlanOnce();
        ASSERT_NE(FindForwardBatch(ready), nullptr);
        const auto& ids = FindForwardBatch(ready)->request_ids;
        ASSERT_LE(ids.size(), 2u);
        served |= std::find(ids.begin(), ids.end(), "b") != ids.end();
        Complete(ready);
    }
    EXPECT_TRUE(served);
}

}  // namespace tokenspeed::test
