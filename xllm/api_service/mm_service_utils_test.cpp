/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "api_service/mm_service_utils.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace xllm {
namespace {

class FakeCall {
 public:
  bool finish_with_error(const StatusCode& code, const std::string& message) {
    called = true;
    status_code = code;
    error_message = message;
    return true;
  }

  bool called = false;
  StatusCode status_code = StatusCode::OK;
  std::string error_message;
};

}  // namespace

TEST(MMServiceUtilsTest, BuildMessagesPreservesToolMetadata) {
  proto::MMChatRequest request;
  auto* req_msg = request.add_messages();
  req_msg->set_role("assistant");
  req_msg->set_tool_call_id("call_0");

  auto* content = req_msg->add_content();
  content->set_type("text");
  content->set_text("I will check.");

  auto* tool_call = req_msg->add_tool_calls();
  tool_call->set_id("functions.get_weather:0");
  tool_call->set_type("function");
  tool_call->mutable_function()->set_name("get_weather");
  tool_call->mutable_function()->set_arguments(R"({"city":"Beijing"})");

  std::vector<Message> messages;
  auto call = std::make_shared<FakeCall>();
  ASSERT_TRUE(mm_service_utils::build_messages(
      request.messages(), messages, call, /*image_limit=*/4));
  EXPECT_FALSE(call->called);

  ASSERT_EQ(messages.size(), 1);
  EXPECT_EQ(messages[0].role, "assistant");

  ASSERT_TRUE(std::holds_alternative<MMContentVec>(messages[0].content));
  const auto& mm_content = std::get<MMContentVec>(messages[0].content);
  ASSERT_EQ(mm_content.size(), 1);
  EXPECT_EQ(mm_content[0].type, "text");
  EXPECT_EQ(mm_content[0].text, "I will check.");

  ASSERT_TRUE(messages[0].tool_call_id.has_value());
  EXPECT_EQ(messages[0].tool_call_id.value(), "call_0");

  ASSERT_TRUE(messages[0].tool_calls.has_value());
  const auto& tool_calls = messages[0].tool_calls.value();
  ASSERT_EQ(tool_calls.size(), 1);
  EXPECT_EQ(tool_calls[0].id, "functions.get_weather:0");
  EXPECT_EQ(tool_calls[0].type, "function");
  EXPECT_EQ(tool_calls[0].function.name, "get_weather");
  EXPECT_EQ(tool_calls[0].function.arguments, R"({"city":"Beijing"})");
}

}  // namespace xllm
