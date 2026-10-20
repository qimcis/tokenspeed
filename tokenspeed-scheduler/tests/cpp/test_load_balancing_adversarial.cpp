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

#include <map>

namespace tokenspeed::test {

class LoadBalancingAdversarialSuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.max_batch_size = 4;
        cfg.max_decode_batch_size = 2;
        cfg.max_scheduled_tokens = 8;
        cfg.device_allocator.total_pages = 256;
        cfg.cache_groups.front().total_pages = 256;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
    void Complete(const ExecutionPlan& plan) {
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        for (const auto& id : FindForwardBatch(plan)->request_ids) SendForwardDone(id, {99});
    }
    void Ready() {
        scheduler_->ConfigureExpertAffinity(4, 2, 32);
        Submit(std::vector<RequestSpec>{{.request_id = "a", .tokens = {1, 2}, .max_new_tokens = 64},
                                        {.request_id = "b", .tokens = {3, 4}, .max_new_tokens = 64},
                                        {.request_id = "c", .tokens = {5, 6}, .max_new_tokens = 64}});
        Complete(PlanOnce());
        Complete(PlanOnce());  // a,b
        Complete(PlanOnce());  // c,a; next ready order is b,c,a
    }
    void Hint(const std::string& id, std::vector<std::uint64_t> keys) {
        ASSERT_TRUE(scheduler_->UpdateExpertRoute(id, scheduler_->RequestGeneration(id),
                                                  scheduler_->RequestTokenSize(id) - 1, keys));
    }
};

TEST_F(LoadBalancingAdversarialSuite, RepeatedExpertWorkIsChargedForEveryRequest) {
    Ready();
    Hint("a", {1});
    Hint("b", {1});
    Hint("c", {2, 3});
    // b+c has three distinct experts but peak work 1; b+a has one
    // distinct expert but peak work 2. Deduplicating the batch is wrong.
    scheduler_->SetExpertPlacement(1, {1, 2, 3}, {{1, 0}, {0, 0.5}, {0, 0.5}});
    const auto plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(plan), nullptr);
    EXPECT_EQ(FindForwardBatch(plan)->request_ids, (std::vector<std::string>{"b", "c"}));
}

TEST_F(LoadBalancingAdversarialSuite, ReplicaWeightsChangeChoiceWithoutReplacingLogicalHints) {
    Ready();
    Hint("a", {1});
    Hint("b", {1});
    Hint("c", {2});
    scheduler_->SetExpertPlacement(1, {1, 2}, {{1, 0}, {0, 1}});
    const auto generation = scheduler_->RequestGeneration("a");
    // Under epoch 1 b+c wins; replicated key 1 in epoch 2 makes b+a win.
    scheduler_->SetExpertPlacement(2, {2, 1}, {{0, 1}, {0.5, 0.5}});
    EXPECT_EQ(scheduler_->RequestGeneration("a"), generation);
    EXPECT_FALSE(scheduler_->UpdateExpertRoute("a", generation, scheduler_->RequestTokenSize("a") - 1, {2}));
    EXPECT_THROW(scheduler_->SetExpertPlacement(1, {1, 2}, {{1, 0}, {0, 1}}), std::invalid_argument);
    const auto plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(plan), nullptr);
    EXPECT_EQ(FindForwardBatch(plan)->request_ids, (std::vector<std::string>{"b", "a"}));
}

TEST_F(LoadBalancingAdversarialSuite, InvalidTopologyEpochCannotReplaceWorkingPlacement) {
    Ready();
    Hint("a", {1});
    Hint("b", {1});
    Hint("c", {2});
    scheduler_->SetExpertPlacement(1, {1, 2}, {{1, 0}, {0, 1}});
    EXPECT_THROW(scheduler_->SetExpertPlacement(2, {1, 2}, {{1}, {1}}), std::invalid_argument);
    EXPECT_EQ(scheduler_->ExpertPlacementEpoch(), 1u);
    const auto plan = PlanOnce();
    ASSERT_NE(FindForwardBatch(plan), nullptr);
    EXPECT_EQ(FindForwardBatch(plan)->request_ids, (std::vector<std::string>{"b", "c"}));
}

TEST_F(LoadBalancingAdversarialSuite, TinyQuotaPreservesFifoAndResetsForEachRound) {
    scheduler_->ConfigurePrefillWorkBudget(0.01, 1, 1, 1);
    Submit(std::vector<RequestSpec>{{.request_id = "a", .tokens = {1, 2, 3}, .max_new_tokens = 2},
                                    {.request_id = "b", .tokens = {4, 5}, .max_new_tokens = 2}});
    const std::vector<std::string> ids{"a", "a", "a", "b", "b"};
    const std::vector<double> costs{2, 3, 4, 2, 3};
    for (std::size_t i = 0; i < ids.size(); ++i) {
        auto plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        EXPECT_EQ(FindForwardBatch(plan)->request_ids, (std::vector<std::string>{ids[i]}));
        EXPECT_EQ(FindForwardBatch(plan)->input_lengths, (std::vector<std::int32_t>{1}));
        EXPECT_DOUBLE_EQ(scheduler_->LastPrefillWorkMs(), costs[i]);
        SendForwardDone(ids[i], (i == 2 || i == 4) ? std::vector<std::int32_t>{99} : std::vector<std::int32_t>{});
    }
    EXPECT_EQ(scheduler_->PrefillWorkOverruns(), 5u);
    Complete(PlanOnce());
    EXPECT_DOUBLE_EQ(scheduler_->LastPrefillWorkMs(), 0);
}

TEST_F(LoadBalancingAdversarialSuite, LargeWorkQuotaCannotExceedHardTokenCap) {
    scheduler_->ConfigurePrefillWorkBudget(1e9, 1, 0, 0);
    Submit(RequestSpec{.request_id = "a", .tokens = MakeTokens(17, 1), .max_new_tokens = 2});
    for (int count : {8, 8, 1}) {
        const auto plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        EXPECT_EQ(FindForwardBatch(plan)->input_lengths, (std::vector<std::int32_t>{count}));
        EXPECT_DOUBLE_EQ(scheduler_->LastPrefillWorkMs(), count);
        SendForwardDone("a", count == 1 ? std::vector<std::int32_t>{99} : std::vector<std::int32_t>{});
    }
}

TEST_F(LoadBalancingAdversarialSuite, FinishingChunkSharesOnlyRemainingWorkWithNextRequest) {
    scheduler_->ConfigurePrefillWorkBudget(10, 1, 1, 0);
    Submit(std::vector<RequestSpec>{{.request_id = "a", .tokens = MakeTokens(11, 1), .max_new_tokens = 2},
                                    {.request_id = "b", .tokens = {21, 22}, .max_new_tokens = 2}});
    auto first = PlanOnce();
    ASSERT_NE(FindForwardBatch(first), nullptr);
    EXPECT_EQ(FindForwardBatch(first)->input_lengths, (std::vector<std::int32_t>{8}));
    SendForwardDone("a", {});
    for (int prefix : {8, 9, 10}) {
        const auto plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        EXPECT_EQ(FindForwardBatch(plan)->request_ids.front(), "a");
        EXPECT_EQ(FindForwardBatch(plan)->input_lengths.front(), 1);
        // At prefix 10 even a single row costs 11: no second fallback is
        // allowed in that round for the waiting b request.
        EXPECT_EQ(FindForwardBatch(plan)->request_ids.size(), 1u);
        EXPECT_DOUBLE_EQ(scheduler_->LastPrefillWorkMs(), prefix + 1);
        SendForwardDone("a", prefix == 10 ? std::vector<std::int32_t>{99} : std::vector<std::int32_t>{});
    }
    EXPECT_EQ(scheduler_->PrefillWorkOverruns(), 1u);
    const auto last = PlanOnce();
    ASSERT_NE(FindForwardBatch(last), nullptr);
    EXPECT_EQ(FindForwardBatch(last)->request_ids, (std::vector<std::string>{"b"}));
    EXPECT_DOUBLE_EQ(scheduler_->LastPrefillWorkMs(), 2);
}

TEST_F(LoadBalancingAdversarialSuite, PlacementUpdatesCannotStarveExpensiveResident) {
    Ready();
    std::map<std::string, int> serviced;
    for (int round = 0; round < 30; ++round) {
        // Refresh only completed frontiers; unchanged frontiers may reject a
        // duplicate observation while their prior logical hint stays valid.
        for (const auto& id : {"a", "b", "c"}) {
            scheduler_->UpdateExpertRoute(id, scheduler_->RequestGeneration(id), scheduler_->RequestTokenSize(id) - 1,
                                          {id == std::string("c") ? 2u : 1u});
        }
        scheduler_->SetExpertPlacement(round + 1, {1, 2},
                                       round % 2 == 0 ? std::vector<std::vector<double>>{{1, 0}, {100, 0}}
                                                      : std::vector<std::vector<double>>{{0, 1}, {0, 100}});
        const auto plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        ASSERT_LE(FindForwardBatch(plan)->request_ids.size(), 2u);
        for (const auto& id : FindForwardBatch(plan)->request_ids) ++serviced[id];
        Complete(plan);
    }
    // max_deferrals + cap bounds service gaps to four productive rounds.
    for (const auto& id : {"a", "b", "c"}) EXPECT_GE(serviced[id], 7);
}

}  // namespace tokenspeed::test
