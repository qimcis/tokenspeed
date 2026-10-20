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

namespace tokenspeed::test {

TEST(ExpertAffinityPolicy, UnionTiesUnknownBarrierAndForcedService) {
    const std::vector<std::uint64_t> a{1}, b{2}, c{1};
    std::vector<ExpertAffinityCandidate> candidates{{a, 0}, {b, 0}, {c, 0}};
    EXPECT_EQ(SelectExpertAffinity(candidates, {}, 2), 0u);
    EXPECT_EQ(SelectExpertAffinity(candidates, {1}, 2), 0u);
    EXPECT_EQ(SelectExpertAffinity(std::span(candidates).subspan(1), {1}, 2), 1u);
    candidates[1].deferrals = 2;
    EXPECT_EQ(SelectExpertAffinity(candidates, {1}, 2), 1u);
    candidates[0].keys = {};
    EXPECT_EQ(SelectExpertAffinity(candidates, {1}, 2), 0u);
}

class ExpertAffinitySuite : public SchedulerTestSuite {
protected:
    SchedulerConfig MakeConfig() override {
        auto cfg = SchedulerTestSuite::MakeConfig();
        cfg.max_batch_size = 4;
        cfg.max_decode_batch_size = 2;
        cfg.device_allocator.total_pages = 256;
        cfg.cache_groups.front().total_pages = 256;
        cfg.disable_l2_cache = true;
        cfg.disable_prefix_cache = true;
        SetTestSnapshotPool(cfg);
        return cfg;
    }
    void Ready(bool enable) {
        if (enable) scheduler_->ConfigureExpertAffinity(4, 1, 8);
        Submit(std::vector<RequestSpec>{{.request_id = "a", .tokens = {1, 2}, .max_new_tokens = 32},
                                        {.request_id = "b", .tokens = {3, 4}, .max_new_tokens = 32},
                                        {.request_id = "c", .tokens = {5, 6}, .max_new_tokens = 32}});
        auto prefill = PlanOnce();
        ASSERT_EQ(FindForwardBatch(prefill)->request_ids.size(), 3u);
        for (const auto& id : FindForwardBatch(prefill)->request_ids) SendForwardDone(id, {10});
        auto bootstrap = PlanOnce();
        ASSERT_EQ(FindForwardBatch(bootstrap)->request_ids, (std::vector<std::string>{"a", "b"}));
        Complete(bootstrap);
        auto next = PlanOnce();
        ASSERT_EQ(FindForwardBatch(next)->request_ids, (std::vector<std::string>{"c", "a"}));
        Complete(next);
    }
    void Complete(const ExecutionPlan& plan) {
        for (const auto& id : FindForwardBatch(plan)->request_ids) SendForwardDone(id, {11});
    }
    bool Hint(const std::string& id, std::uint64_t key) {
        return scheduler_->UpdateExpertRoute(id, scheduler_->RequestGeneration(id),
                                             scheduler_->RequestTokenSize(id) - 1, {key});
    }
};

TEST_F(ExpertAffinitySuite, RealSchedulerChoosesSmallerUnionThenForcesDeferredRequest) {
    Ready(true);
    ASSERT_TRUE(Hint("a", 1));
    ASSERT_TRUE(Hint("b", 1));
    ASSERT_TRUE(Hint("c", 2));
    auto first = PlanOnce();
    ASSERT_EQ(FindForwardBatch(first)->request_ids, (std::vector<std::string>{"b", "a"}));
    const auto& batch = *FindForwardBatch(first);
    EXPECT_EQ(batch.request_generations[0], scheduler_->RequestGeneration("b"));
    EXPECT_EQ(batch.computed_frontiers[0], scheduler_->RequestTokenSize("b"));
    Complete(first);
    auto second = PlanOnce();
    EXPECT_EQ(FindForwardBatch(second)->request_ids.front(), "c");
}

TEST_F(ExpertAffinitySuite, DisabledAndUnknownUseSameFifoCap) {
    Ready(false);
    auto plan = PlanOnce();
    EXPECT_EQ(FindForwardBatch(plan)->request_ids, (std::vector<std::string>{"b", "c"}));
    EXPECT_FALSE(Hint("a", 1));
}

TEST_F(ExpertAffinitySuite, UnknownHintIsFifoBarrier) {
    Ready(true);
    ASSERT_TRUE(Hint("a", 1));
    ASSERT_TRUE(Hint("c", 1));
    auto plan = PlanOnce();
    EXPECT_EQ(FindForwardBatch(plan)->request_ids, (std::vector<std::string>{"b", "c"}));
}

TEST_F(ExpertAffinitySuite, RejectsStaleFutureDuplicateInflightAndReusedIdentity) {
    Ready(true);
    const auto generation = scheduler_->RequestGeneration("a");
    ASSERT_TRUE(Hint("a", 1));
    EXPECT_FALSE(Hint("a", 2));
    EXPECT_FALSE(scheduler_->UpdateExpertRoute("a", generation, 999, {1}));
    auto plan = PlanOnce();
    EXPECT_FALSE(scheduler_->UpdateExpertRoute("a", generation, 4, {1}));
    Complete(plan);
    SendAbortEvent("a");
    PlanOnce();
    Submit(RequestSpec{.request_id = "a", .tokens = {21, 22}, .max_new_tokens = 32});
    EXPECT_NE(scheduler_->RequestGeneration("a"), generation);
    EXPECT_FALSE(scheduler_->UpdateExpertRoute("a", generation, 3, {1}));
}

TEST_F(ExpertAffinitySuite, RejectsObservationOutsideTokenAgeBudget) {
    scheduler_->ConfigureExpertAffinity(4, 1, 1);
    Ready(false);
    ASSERT_TRUE(Hint("a", 1));
    ASSERT_TRUE(Hint("b", 1));
    ASSERT_TRUE(Hint("c", 2));
    const auto old_frontier = scheduler_->RequestTokenSize("a") - 1;
    for (int i = 0; i < 8; ++i) Complete(PlanOnce());
    ASSERT_GT(scheduler_->RequestTokenSize("a") - 1 - (old_frontier + 1), 1);
    EXPECT_FALSE(scheduler_->UpdateExpertRoute("a", scheduler_->RequestGeneration("a"), old_frontier + 1, {1}));
}

TEST_F(ExpertAffinitySuite, UnsupportedModesRejectAffinityEarly) {
    auto cfg = MakeConfig();
    cfg.overlap_schedule_depth = 1;
    Scheduler overlap(cfg);
    EXPECT_THROW(overlap.ConfigureExpertAffinity(4, 1, 8), std::invalid_argument);
    cfg.overlap_schedule_depth = 0;
    cfg.enable_mixed_prefill_decode = true;
    Scheduler mixed(cfg);
    EXPECT_THROW(mixed.ConfigureExpertAffinity(4, 1, 8), std::invalid_argument);
    cfg.enable_mixed_prefill_decode = false;
    cfg.decode_input_tokens = 2;
    Scheduler speculative(cfg);
    EXPECT_THROW(speculative.ConfigureExpertAffinity(4, 1, 8), std::invalid_argument);
}

TEST_F(ExpertAffinitySuite, PlacementBalancesRankWorkAndRetainsLogicalHintsAcrossEpochs) {
    Ready(true);
    ASSERT_TRUE(Hint("a", 1));
    ASSERT_TRUE(Hint("b", 1));
    ASSERT_TRUE(Hint("c", 2));
    scheduler_->SetExpertPlacement(1, {1, 2}, {{1, 0}, {0, 1}});
    auto first = PlanOnce();
    EXPECT_EQ(FindForwardBatch(first)->request_ids, (std::vector<std::string>{"b", "c"}));
    EXPECT_THROW(scheduler_->SetExpertPlacement(2, {1, 2}, {{0, 1}, {1, 0}}), std::invalid_argument);
    Complete(first);
    const auto generation = scheduler_->RequestGeneration("a");
    scheduler_->SetExpertPlacement(2, {1, 2}, {{0.5, 0.5}, {1, 0}});
    EXPECT_EQ(scheduler_->RequestGeneration("a"), generation);
    EXPECT_EQ(scheduler_->ExpertPlacementEpoch(), 2u);
    EXPECT_THROW(scheduler_->SetExpertPlacement(2, {1, 2}, {{1, 0}, {0, 1}}), std::invalid_argument);
    EXPECT_THROW(scheduler_->SetExpertPlacement(3, {1}, {{1, 0}}), std::invalid_argument);
    EXPECT_EQ(scheduler_->ExpertPlacementEpoch(), 2u);
    auto next = PlanOnce();
    EXPECT_EQ(FindForwardBatch(next)->request_ids.front(), "a");
}

TEST_F(ExpertAffinitySuite, PlacementMissingKeysRemainFifoBarriers) {
    Ready(true);
    ASSERT_TRUE(Hint("a", 1));
    ASSERT_TRUE(Hint("b", 1));
    ASSERT_TRUE(Hint("c", 9));
    scheduler_->SetExpertPlacement(1, {1, 2}, {{1, 0}, {0, 1}});
    auto plan = PlanOnce();
    EXPECT_EQ(FindForwardBatch(plan)->request_ids, (std::vector<std::string>{"b", "c"}));
}

TEST_F(ExpertAffinitySuite, WorkBudgetSharedAndTinyBudgetProgress) {
    scheduler_->ConfigurePrefillWorkBudget(3, 1, 0, 0);
    Submit(std::vector<RequestSpec>{{.request_id = "a", .tokens = {1, 2}, .max_new_tokens = 2},
                                    {.request_id = "b", .tokens = {3, 4, 5}, .max_new_tokens = 2}});
    auto plan = PlanOnce();
    EXPECT_EQ(FindForwardBatch(plan)->input_lengths, (std::vector<std::int32_t>{2, 1}));
    EXPECT_DOUBLE_EQ(scheduler_->LastPrefillWorkMs(), 3);
    EXPECT_EQ(scheduler_->PrefillWorkOverruns(), 0u);
}

TEST_F(ExpertAffinitySuite, TinyBudgetNeverStallsAndContextChangesQuota) {
    scheduler_->ConfigurePrefillWorkBudget(0.01, 1, 1, 1);
    Submit(RequestSpec{.request_id = "a", .tokens = {1, 2, 3}, .max_new_tokens = 2});
    for (int i = 0; i < 3; ++i) {
        auto plan = PlanOnce();
        ASSERT_NE(FindForwardBatch(plan), nullptr);
        EXPECT_EQ(FindForwardBatch(plan)->input_lengths, (std::vector<std::int32_t>{1}));
        EXPECT_DOUBLE_EQ(scheduler_->LastPrefillWorkMs(), 2 + i);
        SendForwardDone("a", i == 2 ? std::vector<std::int32_t>{11} : std::vector<std::int32_t>{});
    }
    EXPECT_EQ(scheduler_->PrefillWorkOverruns(), 3u);
}

TEST_F(ExpertAffinitySuite, RejectsLateConfigurationAndUnboundedWindows) {
    EXPECT_THROW(scheduler_->ConfigureExpertAffinity(2, 1, 8), std::invalid_argument);
    EXPECT_THROW(scheduler_->ConfigureExpertAffinity(257, 1, 8), std::invalid_argument);
    Ready(true);
    EXPECT_THROW(scheduler_->ConfigureExpertAffinity(4, 1, 8), std::invalid_argument);
}

}  // namespace tokenspeed::test
