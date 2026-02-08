#include <gtest/gtest.h>

#include <sstream>

#include "utils.h"

// --- split_string ---

TEST(Utils, SplitStringBasic) {
    auto tokens = Utils::split_string("a,b,c", ',');
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], "a");
    EXPECT_EQ(tokens[1], "b");
    EXPECT_EQ(tokens[2], "c");
}

TEST(Utils, SplitStringEmpty) {
    auto tokens = Utils::split_string("", ',');
    EXPECT_EQ(tokens.size(), 0u);
}

TEST(Utils, SplitStringNoDelimiter) {
    auto tokens = Utils::split_string("hello", ',');
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0], "hello");
}

TEST(Utils, SplitStringTrailingDelimiter) {
    auto tokens = Utils::split_string("a,b,", ',');
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], "a");
    EXPECT_EQ(tokens[1], "b");
}

TEST(Utils, SplitStringLeadingDelimiter) {
    auto tokens = Utils::split_string(",a", ',');
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], "");
    EXPECT_EQ(tokens[1], "a");
}

TEST(Utils, SplitStringConsecutiveDelimiters) {
    auto tokens = Utils::split_string("a,,b", ',');
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], "a");
    EXPECT_EQ(tokens[1], "");
    EXPECT_EQ(tokens[2], "b");
}

TEST(Utils, SplitStringSpaceDelimiter) {
    auto tokens = Utils::split_string("e2e4 d7d5 e4d5", ' ');
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], "e2e4");
    EXPECT_EQ(tokens[1], "d7d5");
    EXPECT_EQ(tokens[2], "e4d5");
}

// --- trim ---

TEST(Utils, TrimEmpty) { EXPECT_EQ(Utils::trim(""), ""); }

TEST(Utils, TrimNoWhitespace) { EXPECT_EQ(Utils::trim("hello"), "hello"); }

TEST(Utils, TrimLeadingSpaces) { EXPECT_EQ(Utils::trim("   hello"), "hello"); }

TEST(Utils, TrimTrailingSpaces) { EXPECT_EQ(Utils::trim("hello   "), "hello"); }

TEST(Utils, TrimBothSides) { EXPECT_EQ(Utils::trim("  hello world  "), "hello world"); }

TEST(Utils, TrimAllWhitespace) { EXPECT_EQ(Utils::trim("   "), ""); }

TEST(Utils, TrimTabsAndNewlines) { EXPECT_EQ(Utils::trim("\t\n hello \n\t"), "hello"); }

TEST(Utils, TrimSingleChar) { EXPECT_EQ(Utils::trim(" x "), "x"); }

// --- string_ends_with ---

TEST(Utils, EndsWithMatch) {
    EXPECT_TRUE(Utils::string_ends_with("hello.txt", ".txt"));
}

TEST(Utils, EndsWithNoMatch) {
    EXPECT_FALSE(Utils::string_ends_with("hello.txt", ".bin"));
}

TEST(Utils, EndsWithEmptySuffix) { EXPECT_TRUE(Utils::string_ends_with("hello", "")); }

TEST(Utils, EndsWithEmptyString) { EXPECT_FALSE(Utils::string_ends_with("", ".txt")); }

TEST(Utils, EndsWithBothEmpty) { EXPECT_TRUE(Utils::string_ends_with("", "")); }

TEST(Utils, EndsWithSuffixLongerThanString) {
    EXPECT_FALSE(Utils::string_ends_with("hi", "hello"));
}

TEST(Utils, EndsWithExactMatch) {
    EXPECT_TRUE(Utils::string_ends_with("test", "test"));
}

TEST(Utils, EndsWithSingleChar) {
    EXPECT_TRUE(Utils::string_ends_with("abc", "c"));
    EXPECT_FALSE(Utils::string_ends_with("abc", "b"));
}

// --- log functions ---

TEST(Utils, LogInfoWritesToStdout) {
    std::ostringstream buffer;
    std::streambuf* old_buf = std::cout.rdbuf(buffer.rdbuf());
    Utils::log_info("test_message_42");
    std::cout.rdbuf(old_buf);

    std::string output = buffer.str();
    EXPECT_NE(output.find("[INFO"), std::string::npos);
    EXPECT_NE(output.find("test_message_42"), std::string::npos);
}

TEST(Utils, LogErrorWritesToStderr) {
    std::ostringstream buffer;
    std::streambuf* old_buf = std::cerr.rdbuf(buffer.rdbuf());
    Utils::log_error("error_msg_99");
    std::cerr.rdbuf(old_buf);

    std::string output = buffer.str();
    EXPECT_NE(output.find("[ERROR"), std::string::npos);
    EXPECT_NE(output.find("error_msg_99"), std::string::npos);
}

TEST(Utils, LogWarningWritesToStdout) {
    std::ostringstream buffer;
    std::streambuf* old_buf = std::cout.rdbuf(buffer.rdbuf());
    Utils::log_warning("warn_msg_77");
    std::cout.rdbuf(old_buf);

    std::string output = buffer.str();
    EXPECT_NE(output.find("[WARNING"), std::string::npos);
    EXPECT_NE(output.find("warn_msg_77"), std::string::npos);
}

TEST(Utils, LogInfoContainsTimestamp) {
    std::ostringstream buffer;
    std::streambuf* old_buf = std::cout.rdbuf(buffer.rdbuf());
    Utils::log_info("ts_check");
    std::cout.rdbuf(old_buf);

    std::string output = buffer.str();
    // Timestamp format: YYYY-MM-DD HH:MM:SS — look for date pattern
    EXPECT_NE(output.find("202"), std::string::npos);  // Year 202x
    EXPECT_NE(output.find(":"), std::string::npos);    // Time separator
}
