#include <gtest/gtest.h>

#include <limits>
#include <numeric>

#include <ATen/ATen.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/util.h>

using torch::ParamCommsDebugInfo;
using namespace torch::profiler::impl;

namespace {

std::shared_ptr<ParamCommsDebugInfo> makeDebugInfo(
    const std::string& pgName,
    const std::string& pgDesc,
    int rank,
    std::string collName,
    int worldSize,
    int64_t seqNumber,
    bool isP2P,
    int globalRankStart = 0,
    int globalRankStride = 1,
    std::vector<int64_t> inSplitSizes = {},
    std::vector<int64_t> outSplitSizes = {},
    bool isAsync = true) {
  auto info = std::make_shared<ParamCommsDebugInfo>(
      std::make_tuple(pgName, pgDesc),
      rank,
      std::move(collName),
      /*inNelems=*/1024,
      /*outNelems=*/1024,
      /*dType=*/at::kFloat,
      std::move(inSplitSizes),
      std::move(outSplitSizes),
      globalRankStart,
      globalRankStride,
      worldSize,
      isAsync);
  info->setSequenceInfo(seqNumber, isP2P);
  return info;
}

// Helper to get comms_id from saveNcclMeta for a given ParamCommsDebugInfo.
std::string getCommsIdViaSaveNcclMeta(
    const std::shared_ptr<ParamCommsDebugInfo>& debugInfo) {
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PARAM_COMMS_INFO, debugInfo);
  at::RecordFunction fn(at::RecordScope::USER_SCOPE);
  fn._setAsync();
  auto meta = saveNcclMeta(fn);
  if (meta.count(kCommsId) == 0) {
    return "";
  }
  return meta.at(kCommsId);
}

std::unordered_map<std::string, std::string> expectedMetadataMap(
    const std::shared_ptr<ParamCommsDebugInfo>& info,
    bool truncate) {
  auto formatList = [truncate](const std::vector<int64_t>& list) {
    if (truncate && list.size() > 30) {
      return fmt::format(
          "\"[{}, ..., {}]\"",
          fmt::join(list.begin(), list.begin() + 29, ", "),
          list.back());
    }
    return fmt::format("\"[{}]\"", fmt::join(list, ", "));
  };

  std::unordered_map<std::string, std::string> expected{
      {kCommsName, fmt::format("\"{}\"", info->getCollectiveName())},
      {kDtype, fmt::format("\"{}\"", c10::toString(info->getDType()))},
      {kInMsgNelems, std::to_string(info->getInMessageNelems())},
      {kOutMsgNelems, std::to_string(info->getOutMessageNelems())},
      {kInSplit, formatList(info->getInputSplitSizes())},
      {kOutSplit, formatList(info->getOutputSplitSizes())},
      {kGroupSize, std::to_string(info->getWorldSize())},
      {kGroupRanks, formatList(info->getGroupRanks())},
      {kRank, std::to_string(info->getRank())}};
  if (info->getGlobalRankStart() >= 0) {
    expected.emplace(
        kGlobalRankStart, std::to_string(info->getGlobalRankStart()));
  }
  if (info->getGlobalRankStride() > 0) {
    expected.emplace(
        kGlobalRankStride, std::to_string(info->getGlobalRankStride()));
  }
  if (!info->getProcessGroupName().empty()) {
    expected.emplace(
        kProcessGroupName,
        fmt::format("\"{}\"", info->getProcessGroupName()));
  }
  if (!info->getProcessGroupDesc().empty()) {
    expected.emplace(
        kProcessGroupDesc,
        fmt::format("\"{}\"", info->getProcessGroupDesc()));
  }
  if (info->getSequenceNumber() >= 0) {
    expected.emplace(kSeqNum, std::to_string(info->getSequenceNumber()));
    expected.emplace(
        kCommsId,
        std::to_string(c10::get_hash(
            info->getProcessGroupName(),
            info->getSequenceNumber(),
            info->getIsP2P(),
            info->getGlobalRankStart(),
            info->getGlobalRankStride(),
            info->getWorldSize())));
  }
  return expected;
}

thread_local std::optional<std::string> capturedInputTensorStarts;
thread_local std::optional<std::string> capturedOutputTensorStarts;
thread_local SaveNcclMetaConfig activeSaveConfig;
thread_local std::unordered_map<std::string, std::string> capturedSaveMetadata;

std::unique_ptr<at::ObserverContext> captureTensorStarts(
    const at::RecordFunction& fn) {
  capturedInputTensorStarts = captureNcclInputTensorStarts(fn, true);
  capturedOutputTensorStarts = captureNcclOutputTensorStarts(fn, true);
  return nullptr;
}

std::unique_ptr<at::ObserverContext> captureSaveMetadata(
    const at::RecordFunction& fn) {
  capturedSaveMetadata = saveNcclMeta(fn, activeSaveConfig);
  return nullptr;
}

void emptyRecordFunctionEnd(
    const at::RecordFunction&,
    at::ObserverContext*) {}

} // namespace

TEST(CollectiveMetadataTest, CollectsRequiredAndConditionalFields) {
  auto info = makeDebugInfo(
      "pg_uid_123", "default_pg", 2, "send", 4, 42, true, 10, 2);
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PARAM_COMMS_INFO, info);
  at::RecordFunction fn(at::RecordScope::USER_SCOPE);

  auto metadata = collectNcclMeta(fn, true);
  ASSERT_TRUE(metadata.has_value());
  EXPECT_EQ(metadata->collective_name, "send");
  EXPECT_EQ(metadata->dtype, at::kFloat);
  EXPECT_EQ(metadata->in_msg_nelems, 1024);
  EXPECT_EQ(metadata->out_msg_nelems, 1024);
  EXPECT_EQ(metadata->group_size, 4);
  EXPECT_EQ(metadata->rank, 2);
  EXPECT_TRUE(metadata->is_async);
  EXPECT_EQ(metadata->global_rank_start, 10);
  EXPECT_EQ(metadata->global_rank_stride, 2);
  EXPECT_EQ(metadata->process_group_name, "pg_uid_123");
  EXPECT_EQ(metadata->process_group_desc, "default_pg");
  EXPECT_EQ(metadata->p2p_dst, 14);
  EXPECT_FALSE(metadata->p2p_src.has_value());
  EXPECT_EQ(metadata->sequence_number, 42);
  EXPECT_TRUE(metadata->comms_id.has_value());
}

TEST(CollectiveMetadataTest, OmitsAbsentConditionalFields) {
  auto info = makeDebugInfo("", "", 0, "allreduce", 4, -1, false, -1, 0);
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PARAM_COMMS_INFO, info);
  at::RecordFunction fn(at::RecordScope::USER_SCOPE);

  auto metadata = collectNcclMeta(fn, true);
  ASSERT_TRUE(metadata.has_value());
  EXPECT_FALSE(metadata->global_rank_start.has_value());
  EXPECT_FALSE(metadata->global_rank_stride.has_value());
  EXPECT_FALSE(metadata->process_group_name.has_value());
  EXPECT_FALSE(metadata->process_group_desc.has_value());
  EXPECT_FALSE(metadata->p2p_src.has_value());
  EXPECT_FALSE(metadata->p2p_dst.has_value());
  EXPECT_FALSE(metadata->sequence_number.has_value());
  EXPECT_FALSE(metadata->comms_id.has_value());
}

TEST(CollectiveMetadataTest, CollectsP2PSourceRank) {
  auto info =
      makeDebugInfo("pg", "desc", 3, "recv", 4, 7, true, 5, 3);
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PARAM_COMMS_INFO, info);
  at::RecordFunction fn(at::RecordScope::USER_SCOPE);

  auto metadata = collectNcclMeta(fn, true);
  ASSERT_TRUE(metadata.has_value());
  EXPECT_EQ(metadata->p2p_src, 14);
  EXPECT_FALSE(metadata->p2p_dst.has_value());
}

TEST(CollectiveMetadataTest, PreservesListTruncation) {
  std::vector<int64_t> list(31);
  std::iota(list.begin(), list.end(), 0);
  auto info = makeDebugInfo(
      "pg", "desc", 0, "allreduce", 31, 1, false, 0, 1, list, list);
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PARAM_COMMS_INFO, info);
  at::RecordFunction fn(at::RecordScope::USER_SCOPE);

  auto bounded = collectNcclMeta(fn, true);
  ASSERT_TRUE(bounded.has_value());
  EXPECT_EQ(bounded->input_split_sizes.prefix.size(), 29);
  EXPECT_EQ(bounded->input_split_sizes.last, 30);
  EXPECT_EQ(bounded->input_split_sizes.original_size, 31);
  EXPECT_TRUE(bounded->input_split_sizes.truncated);
  EXPECT_EQ(bounded->group_ranks.prefix.size(), 29);
  EXPECT_EQ(
      ncclMetaToLegacyMap(*bounded).at(kInSplit),
      "\"[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, ..., 30]\""); // noqa: B950

  auto full = collectNcclMeta(fn, false);
  ASSERT_TRUE(full.has_value());
  EXPECT_EQ(full->output_split_sizes.prefix, list);
  EXPECT_EQ(full->output_split_sizes.original_size, 31);
  EXPECT_FALSE(full->output_split_sizes.truncated);
  EXPECT_EQ(full->group_ranks.prefix, list);
}

TEST(CollectiveMetadataTest, SerializesMaximumCommsId) {
  auto info = makeDebugInfo("pg", "desc", 0, "allreduce", 1, 1, false);
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PARAM_COMMS_INFO, info);
  at::RecordFunction fn(at::RecordScope::USER_SCOPE);
  auto metadata = collectNcclMeta(fn, true);
  ASSERT_TRUE(metadata.has_value());
  metadata->comms_id = std::numeric_limits<uint64_t>::max();

  EXPECT_EQ(
      ncclMetaToLegacyMap(*metadata).at(kCommsId),
      "18446744073709551615");
}

TEST(CollectiveMetadataTest, MissingDebugInfoReturnsNoMetadata) {
  at::RecordFunction fn(at::RecordScope::USER_SCOPE);
  EXPECT_FALSE(collectNcclMeta(fn, true).has_value());
  EXPECT_TRUE(saveNcclMeta(fn).empty());
}

TEST(CollectiveMetadataTest, TensorStartsRemainOpaque) {
  auto info = makeDebugInfo("pg", "desc", 0, "allreduce", 1, 1, false);
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PARAM_COMMS_INFO, info);
  at::RecordFunction fn(at::RecordScope::USER_SCOPE);
  auto metadata = collectNcclMeta(fn, true);
  ASSERT_TRUE(metadata.has_value());
  const std::string input = R"([[1,{"nested":[2,3]}]])";
  const std::string output = R"([{"args":[[4],5]}])";
  metadata->input_tensor_starts = input;
  metadata->output_tensor_starts = output;

  auto map = ncclMetaToLegacyMap(
      *metadata, SaveNcclMetaConfig{true, true, true, true});
  EXPECT_EQ(map.at(kInTensorsStart), input);
  EXPECT_EQ(map.at(kOutTensorsStart), output);
}

TEST(CollectiveMetadataTest, KinetoEventConvertsTypedTorchOpMetadata) {
  auto info = makeDebugInfo("pg", "desc", 0, "allreduce", 1, 1, false);
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PARAM_COMMS_INFO, info);
  at::RecordFunction fn(at::RecordScope::USER_SCOPE);
  auto metadata = collectNcclMeta(fn, true);
  ASSERT_TRUE(metadata.has_value());
  const auto expected = ncclMetaToLegacyMap(*metadata);

  ExtraFields<EventType::TorchOp> fields{
      TorchOpBasicFields{.name_ = "collective"},
      1,
      1,
      {},
      {},
      {},
      {},
      {},
      std::move(metadata),
      {},
      {},
      false,
      nullptr};
  auto result = Result::create(
      0,
      0,
      torch::profiler::impl::kineto::DeviceAndResource{0, 0},
      std::move(fields));
  torch::autograd::profiler::KinetoEvent event(result, false);
  EXPECT_EQ(event.extraMeta(), expected);
}

TEST(CollectiveMetadataTest, ImportedKernelMetadataRemainsStringMap) {
  extra_meta_t expected{{"kernel metadata", "[1, 2, 3]"}};
  ExtraFields<EventType::Kineto> fields{
      .name_ = "kernel",
      .activity_type_ = libkineto::ActivityType::CONCURRENT_KERNEL,
      .extra_meta_ = expected};
  auto result = Result::create(
      0,
      0,
      torch::profiler::impl::kineto::DeviceAndResource{0, 0},
      std::move(fields));
  torch::autograd::profiler::KinetoEvent event(result, false);
  EXPECT_EQ(event.extraMeta(), expected);
}

TEST(CollectiveMetadataTest, CapturesNestedTensorStarts) {
  set_record_tensor_addrs_enabled_val(true);
  auto callback = at::addThreadLocalCallback(
      at::RecordFunctionCallback(captureTensorStarts, emptyRecordFunctionEnd)
          .needsInputs(true)
          .needsOutputs(true));

  auto first = at::empty({1}, at::kByte);
  auto second = at::empty({1}, at::kByte);
  auto nested = c10::ivalue::Tuple::create(
      {first,
       c10::ivalue::Tuple::create(
           {second, c10::IValue(static_cast<int64_t>(7))})});
  std::vector<c10::IValue> inputs{nested};
  at::RecordFunction fn(at::RecordScope::USER_SCOPE);
  fn.setOutputs(std::vector<c10::IValue>{nested});
  fn.before("collective", c10::ArrayRef<const c10::IValue>(inputs));

  const auto expected = fmt::format(
      "\"[[{}, {}, -1]]\"",
      getTensorStartHint(first),
      getTensorStartHint(second));
  EXPECT_EQ(capturedInputTensorStarts, expected);
  EXPECT_EQ(capturedOutputTensorStarts, expected);
  at::removeCallback(callback);
}

TEST(CollectiveMetadataTest, SaveNcclMetaPreservesEveryConfiguration) {
  std::vector<int64_t> splits(31);
  std::iota(splits.begin(), splits.end(), 0);
  auto info = makeDebugInfo(
      "pg", "desc", 0, "allreduce", 31, 42, false, 0, 1, splits, splits);
  c10::DebugInfoGuard guard(c10::DebugInfoKind::PARAM_COMMS_INFO, info);
  set_record_tensor_addrs_enabled_val(true);
  auto callback = at::addThreadLocalCallback(
      at::RecordFunctionCallback(captureSaveMetadata, emptyRecordFunctionEnd)
          .needsInputs(true)
          .needsOutputs(true));

  for (const bool truncate : {false, true}) {
    for (const bool introspectMetadata : {false, true}) {
      for (const bool introspectInputs : {false, true}) {
        for (const bool introspectOutputs : {false, true}) {
          SaveNcclMetaConfig config{
              truncate,
              introspectMetadata,
              introspectInputs,
              introspectOutputs};
          activeSaveConfig = config;
          std::vector<c10::IValue> inputs;
          at::RecordFunction fn(at::RecordScope::USER_SCOPE);
          fn.setOutputs(std::vector<c10::IValue>{});
          fn.before("collective", c10::ArrayRef<const c10::IValue>(inputs));
          std::unordered_map<std::string, std::string> expected{
              {kIsAsynchronizedOp, "1"}};
          if (introspectMetadata) {
            auto core = expectedMetadataMap(info, truncate);
            expected.insert(core.begin(), core.end());
          }
          if (introspectInputs) {
            expected.emplace(kInTensorsStart, "\"[]\"");
          }
          if (introspectOutputs) {
            expected.emplace(kOutTensorsStart, "\"[]\"");
          }
          EXPECT_EQ(capturedSaveMetadata, expected);
        }
      }
    }
  }
  at::removeCallback(callback);
}

TEST(CommsIdTest, ParamCommsDebugInfoStoresSeqNumberAndIsP2P) {
  auto info =
      makeDebugInfo("pg_uid_123", "default_pg", 0, "allreduce", 8, 42, false);

  EXPECT_EQ(info->getSequenceNumber(), 42);
  EXPECT_FALSE(info->getIsP2P());
  EXPECT_EQ(info->getProcessGroupName(), "pg_uid_123");
  EXPECT_EQ(info->getCollectiveName(), "allreduce");
  EXPECT_EQ(info->getWorldSize(), 8);
}

TEST(CommsIdTest, ParamCommsDebugInfoP2PFlag) {
  auto info = makeDebugInfo("pg_uid_456", "custom_pg", 3, "send", 4, 7, true);

  EXPECT_EQ(info->getSequenceNumber(), 7);
  EXPECT_TRUE(info->getIsP2P());
}

TEST(CommsIdTest, ParamCommsDebugInfoDefaultSeqNumberAndIsP2P) {
  auto info = std::make_shared<ParamCommsDebugInfo>(
      std::make_tuple(std::string("pg_uid_789"), std::string("default_pg")),
      /*rank=*/1,
      /*collName=*/std::string("allgather"),
      /*inNelems=*/256,
      /*outNelems=*/512,
      /*dType=*/at::kFloat,
      /*inSplitSizes=*/std::vector<int64_t>{},
      /*outSplitSizes=*/std::vector<int64_t>{},
      /*globalRankStart=*/0,
      /*globalRankStride=*/1,
      /*worldSize=*/2);

  EXPECT_EQ(info->getSequenceNumber(), -1);
  EXPECT_FALSE(info->getIsP2P());
}

TEST(CommsIdTest, SaveNcclMetaEmitsCommsId) {
  auto debugInfo =
      makeDebugInfo("pg_uid_123", "default_pg", 0, "allreduce", 8, 42, false);

  auto commsId = getCommsIdViaSaveNcclMeta(debugInfo);
  EXPECT_FALSE(commsId.empty());

  // Verify determinism: same input produces the same comms_id
  auto commsId2 = getCommsIdViaSaveNcclMeta(debugInfo);
  EXPECT_EQ(commsId, commsId2);
}

TEST(CommsIdTest, SaveNcclMetaOmitsCommsIdWhenSeqNotSet) {
  auto debugInfo = std::make_shared<ParamCommsDebugInfo>(
      std::make_tuple(std::string("pg_uid_no_seq"), std::string("default_pg")),
      /*rank=*/0,
      /*collName=*/std::string("allreduce"),
      /*inNelems=*/1024,
      /*outNelems=*/1024,
      /*dType=*/at::kFloat,
      /*inSplitSizes=*/std::vector<int64_t>{},
      /*outSplitSizes=*/std::vector<int64_t>{},
      /*globalRankStart=*/0,
      /*globalRankStride=*/1,
      /*worldSize=*/8);

  auto commsId = getCommsIdViaSaveNcclMeta(debugInfo);
  EXPECT_TRUE(commsId.empty());
}

TEST(CommsIdTest, CommsIdDiffersForDifferentSeqNumbers) {
  auto id1 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 8, 1, false));
  auto id2 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 8, 2, false));
  EXPECT_NE(id1, id2);
}

TEST(CommsIdTest, CommsIdDiffersForDifferentPGNames) {
  auto id1 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg_A", "desc", 0, "allreduce", 8, 42, false));
  auto id2 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg_B", "desc", 0, "allreduce", 8, 42, false));
  EXPECT_NE(id1, id2);
}

TEST(CommsIdTest, CommsIdDiffersForP2PvsCollective) {
  auto id_collective = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 8, 42, false));
  auto id_p2p = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "send", 8, 42, true));
  EXPECT_NE(id_collective, id_p2p);
}

TEST(CommsIdTest, CommsIdDiffersForDifferentTopology) {
  auto id1 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 8, 42, false, 0, 1));
  auto id2 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 4, 42, false, 0, 1));
  EXPECT_NE(id1, id2);

  auto id3 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 4, 42, false, 0, 2));
  auto id4 = getCommsIdViaSaveNcclMeta(
      makeDebugInfo("pg", "desc", 0, "allreduce", 4, 42, false, 1, 2));
  EXPECT_NE(id3, id4);
}
