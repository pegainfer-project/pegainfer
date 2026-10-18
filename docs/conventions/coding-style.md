# Coding Style

## Testing

Don't test for the sake of testing. Prefer integration tests over unit tests — if the E2E test catches it, a unit test is ceremony. Unit tests earn their place for silent failures: GPU kernels, tricky data-structure invariants, edge-case-rich pure logic.

## Logging

Log through `pegainfer-core::logging`. The text layout already prints each record's module target, so don't prefix messages with a module or model name — no `kimi-k2:`, no `Qwen3.5 `. Error messages in `anyhow!` / `bail!` keep their prefix; they surface to callers without a target.
