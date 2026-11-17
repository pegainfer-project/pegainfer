// Record format and file for the tuned cublasLt algo store. Deals in eight
// opaque 64-bit words and knows nothing about cublasLt, so it runs on the CPU.
//
//   v1 <key> <w0> ... <w7>\n        each word exactly 16 hex digits, last wins
//
// A regular file may write short — PIPE_BUF atomicity belongs to pipes, not to
// files — so a torn tail is recovered from rather than ruled out. Strict widths
// stop one parsing as a hit, and an append starts a fresh line when the file
// does not end on one so a torn tail cannot swallow the record after it.
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

namespace {

constexpr int kWords = 8;
constexpr size_t kWordHex = 16;
// Comfortably over a record: linear.cu bounds the key at 512 bytes and the
// words add about 140.
constexpr size_t kLineCap = 1024;

bool decode_words(const char *text, size_t len, unsigned long long out[kWords]) {
  size_t off = 0;
  for (int i = 0; i < kWords; ++i) {
    if (i > 0) {
      if (off >= len || text[off] != ' ') {
        return false;
      }
      ++off;
    }
    if (off + kWordHex > len) {
      return false;
    }
    unsigned long long value = 0;
    for (size_t d = 0; d < kWordHex; ++d) {
      const char c = text[off + d];
      int nibble;
      if (c >= '0' && c <= '9') {
        nibble = c - '0';
      } else if (c >= 'a' && c <= 'f') {
        nibble = c - 'a' + 10;
      } else if (c >= 'A' && c <= 'F') {
        nibble = c - 'A' + 10;
      } else {
        return false;
      }
      value = (value << 4) | static_cast<unsigned long long>(nibble);
    }
    out[i] = value;
    off += kWordHex;
  }
  // Anything after the eighth word means this is not the record it looks like.
  return off == len;
}

}  // namespace

extern "C" {

// 1 when a complete, exactly-formed record for `key` was found; the last wins.
int pegainfer_lt_store_lookup(const char *path, const char *key,
                              unsigned long long out[kWords]) {
  if (path == nullptr || key == nullptr || out == nullptr) {
    return 0;
  }
  std::FILE *f = std::fopen(path, "r");
  if (f == nullptr) {
    return 0;
  }
  char prefix[kLineCap];
  const int prefix_len = std::snprintf(prefix, sizeof(prefix), "v1 %s ", key);
  if (prefix_len <= 0 || static_cast<size_t>(prefix_len) >= sizeof(prefix)) {
    std::fclose(f);
    return 0;
  }
  char line[kLineCap];
  int found = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    char *newline = std::strchr(line, '\n');
    if (newline == nullptr) {
      // A short write's tail or an over-long line. Neither is a record; drain
      // the latter so the next iteration starts on a boundary.
      if (std::feof(f) == 0) {
        int c;
        while ((c = std::fgetc(f)) != EOF && c != '\n') {
        }
      }
      continue;
    }
    *newline = '\0';
    const size_t len = static_cast<size_t>(newline - line);
    if (len <= static_cast<size_t>(prefix_len) ||
        std::strncmp(line, prefix, static_cast<size_t>(prefix_len)) != 0) {
      continue;
    }
    unsigned long long words[kWords];
    if (decode_words(line + prefix_len, len - static_cast<size_t>(prefix_len), words)) {
      std::memcpy(out, words, sizeof(words));
      found = 1;
    }
  }
  std::fclose(f);
  return found;
}

// Best effort: losing a record costs the next start a search, so failures are quiet.
void pegainfer_lt_store_append(const char *path, const char *key,
                               const unsigned long long words[kWords]) {
  if (path == nullptr || key == nullptr || words == nullptr) {
    return;
  }
  char line[kLineCap];
  int off = std::snprintf(line, sizeof(line), "v1 %s", key);
  if (off <= 0 || static_cast<size_t>(off) >= sizeof(line)) {
    return;
  }
  for (int i = 0; i < kWords; ++i) {
    const int n = std::snprintf(line + off, sizeof(line) - static_cast<size_t>(off), " %016llx",
                                words[i]);
    if (n <= 0 || static_cast<size_t>(off + n) >= sizeof(line)) {
      return;
    }
    off += n;
  }
  // O_RDWR because the re-sync below has to look at the last byte; O_APPEND
  // still governs where the write lands.
  const int fd = ::open(path, O_RDWR | O_CREAT | O_APPEND, 0644);
  if (fd < 0) {
    return;
  }
  const off_t end = ::lseek(fd, 0, SEEK_END);
  bool resync = false;
  if (end > 0) {
    char last = '\n';
    if (::pread(fd, &last, 1, end - 1) == 1 && last != '\n') {
      resync = true;  // an earlier write was cut short; do not append onto it
    }
  }
  char out[kLineCap + 1];
  int total = 0;
  if (resync) {
    out[total++] = '\n';
  }
  std::memcpy(out + total, line, static_cast<size_t>(off));
  total += off;
  out[total++] = '\n';
  const ssize_t written = ::write(fd, out, static_cast<size_t>(total));
  (void)written;  // the reader skips a torn tail and the next append re-syncs past it
  ::close(fd);
}

}  // extern "C"
