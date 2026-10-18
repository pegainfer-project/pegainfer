#pragma once

#include <cstdint>

namespace vllm {

class ScalarType {
 public:
  using Id = int64_t;

  constexpr ScalarType(Id id, int64_t bits) : id_(id), bits_(bits) {}

  constexpr Id id() const { return id_; }
  constexpr int64_t size_bits() const { return bits_; }

  static constexpr ScalarType from_id(Id id) {
    switch (id) {
      case 1:
        return ScalarType(1, 4);
      case 2:
        return ScalarType(2, 4);
      case 3:
        return ScalarType(3, 8);
      case 4:
        return ScalarType(4, 8);
      case 5:
        return ScalarType(5, 4);
      case 6:
        return ScalarType(6, 8);
      case 7:
        return ScalarType(7, 8);
      case 8:
        return ScalarType(8, 16);
      case 9:
        return ScalarType(9, 16);
      case 10:
        return ScalarType(10, 4);
      case 11:
        return ScalarType(11, 8);
      default:
        return ScalarType(0, 0);
    }
  }

  constexpr bool operator==(ScalarType const& other) const {
    return id_ == other.id_;
  }

  constexpr bool operator!=(ScalarType const& other) const {
    return !(*this == other);
  }

 private:
  Id id_;
  int64_t bits_;
};

using ScalarTypeId = ScalarType::Id;

static inline constexpr auto kU4 = ScalarType(1, 4);
static inline constexpr auto kU4B8 = ScalarType(2, 4);
static inline constexpr auto kU8 = ScalarType(3, 8);
static inline constexpr auto kU8B128 = ScalarType(4, 8);
static inline constexpr auto kFE2M1f = ScalarType(5, 4);
static inline constexpr auto kFE4M3fn = ScalarType(6, 8);
static inline constexpr auto kFE8M0fnu = ScalarType(7, 8);
static inline constexpr auto kFloat16 = ScalarType(8, 16);
static inline constexpr auto kBFloat16 = ScalarType(9, 16);
static inline constexpr auto kS4 = ScalarType(10, 4);
static inline constexpr auto kS8 = ScalarType(11, 8);

}  // namespace vllm
