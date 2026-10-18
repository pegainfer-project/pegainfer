use std::collections::BTreeMap;
use std::env;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use cudarc::driver::sys;
use pegainfer_core::tensor::DeviceContext;
use serde::Serialize;

#[derive(Clone, Debug, Default)]
pub struct DecodeAttributionProfile {
    enabled: bool,
    nvtx_enabled: bool,
    nvtx_range_count: usize,
    total_generation_us: u64,
    prefill_next_token_us: Option<u64>,
    per_token_decode_us: Vec<u64>,
    samples: Vec<SectionSample>,
    gpu_samples: Vec<GpuSectionSample>,
    gpu_timing_failures: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct SectionSample {
    phase: &'static str,
    section: &'static str,
    call_site: String,
    layer: Option<usize>,
    token_index: Option<usize>,
    elapsed_us: u64,
}

#[derive(Clone, Debug, Serialize)]
struct GpuSectionSample {
    phase: &'static str,
    section: &'static str,
    call_site: String,
    layer: Option<usize>,
    token_index: Option<usize>,
    device_ordinals: Vec<usize>,
    elapsed_us: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SectionRollup {
    pub section: String,
    pub calls: usize,
    pub total_us: u64,
    pub mean_us: f64,
    pub min_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
    pub max_us: u64,
    pub pct: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CallSiteRollup {
    call_site: String,
    section: String,
    calls: usize,
    total_us: u64,
    mean_us: f64,
    min_us: u64,
    p50_us: u64,
    p95_us: u64,
    p99_us: u64,
    max_us: u64,
    pct: f64,
}

impl DecodeAttributionProfile {
    pub(crate) fn disabled() -> Self {
        Self::default()
    }

    pub(crate) fn enabled() -> Self {
        Self {
            enabled: true,
            nvtx_enabled: nvtx_enabled_from_env(),
            ..Self::default()
        }
    }

    pub(crate) fn set_total_generation(&mut self, elapsed: Duration) {
        if self.enabled {
            self.total_generation_us = micros(elapsed);
        }
    }

    pub(crate) fn set_prefill_next_token(&mut self, elapsed: Duration) {
        if self.enabled {
            self.prefill_next_token_us = Some(micros(elapsed));
        }
    }

    pub(crate) fn push_decode_token(&mut self, elapsed: Duration) {
        if self.enabled {
            self.per_token_decode_us.push(micros(elapsed));
        }
    }

    pub(crate) fn record_result<T, C, S>(
        &mut self,
        phase: &'static str,
        section: &'static str,
        call_site: C,
        layer: Option<usize>,
        token_index: Option<usize>,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T>
    where
        C: FnOnce() -> S,
        S: Into<String>,
    {
        if !self.enabled {
            return f();
        }
        let start = Instant::now();
        let result = f();
        self.samples.push(SectionSample {
            phase,
            section,
            call_site: call_site().into(),
            layer,
            token_index,
            elapsed_us: micros(start.elapsed()),
        });
        result
    }

    pub(crate) fn record_gpu_result<T, C, S>(
        &mut self,
        ctx: &DeviceContext,
        phase: &'static str,
        section: &'static str,
        call_site: C,
        layer: Option<usize>,
        token_index: Option<usize>,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T>
    where
        C: FnOnce() -> S,
        S: Into<String>,
    {
        self.record_gpu_contexts(&[ctx], phase, section, call_site, layer, token_index, f)
    }

    pub(crate) fn record_gpu_pair_result<T, C, S>(
        &mut self,
        rank0: &DeviceContext,
        rank1: &DeviceContext,
        phase: &'static str,
        section: &'static str,
        call_site: C,
        layer: Option<usize>,
        token_index: Option<usize>,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T>
    where
        C: FnOnce() -> S,
        S: Into<String>,
    {
        self.record_gpu_contexts(
            &[rank0, rank1],
            phase,
            section,
            call_site,
            layer,
            token_index,
            f,
        )
    }

    fn record_gpu_contexts<T, C, S>(
        &mut self,
        contexts: &[&DeviceContext],
        phase: &'static str,
        section: &'static str,
        call_site: C,
        layer: Option<usize>,
        token_index: Option<usize>,
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T>
    where
        C: FnOnce() -> S,
        S: Into<String>,
    {
        if !self.enabled {
            return f();
        }
        assert!(
            !contexts.is_empty(),
            "GPU attribution requires at least one CUDA context"
        );

        let call_site = call_site().into();
        let _nvtx_range = if self.nvtx_enabled {
            let range = nvtx::range!("dsv2lite.ep2.{}", call_site);
            self.nvtx_range_count += 1;
            Some(range)
        } else {
            None
        };

        let timing_state = (|| -> Result<_> {
            let mut start_events = Vec::with_capacity(contexts.len());
            let mut end_events = Vec::with_capacity(contexts.len());
            for ctx in contexts {
                start_events.push(
                    ctx.ctx
                        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?,
                );
                end_events.push(
                    ctx.ctx
                        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?,
                );
            }
            for (ctx, event) in contexts.iter().zip(&start_events) {
                event.record(&ctx.stream)?;
            }
            Ok((start_events, end_events))
        })();

        let cpu_start = Instant::now();
        let result = f();
        let cpu_elapsed_us = micros(cpu_start.elapsed());
        let timing_result = timing_state.and_then(|(start_events, end_events)| {
            let mut elapsed_us = 0.0f64;
            for (ctx, event) in contexts.iter().zip(&end_events) {
                event.record(&ctx.stream)?;
            }
            for (start, end) in start_events.iter().zip(&end_events) {
                elapsed_us = elapsed_us.max(f64::from(start.elapsed_ms(end)?) * 1_000.0);
            }
            Ok(elapsed_us.ceil().min(u64::MAX as f64) as u64)
        });

        match timing_result {
            Ok(elapsed_us) => {
                self.gpu_samples.push(GpuSectionSample {
                    phase,
                    section,
                    call_site: call_site.clone(),
                    layer,
                    token_index,
                    device_ordinals: contexts.iter().map(|ctx| ctx.device_ordinal).collect(),
                    elapsed_us,
                });
            }
            Err(_) => {
                self.gpu_timing_failures += 1;
            }
        }
        self.samples.push(SectionSample {
            phase,
            section,
            call_site,
            layer,
            token_index,
            elapsed_us: cpu_elapsed_us,
        });

        result
    }

    pub fn total_generation_us(&self) -> u64 {
        self.total_generation_us
    }

    pub fn prefill_next_token_us(&self) -> Option<u64> {
        self.prefill_next_token_us
    }

    pub fn per_token_decode_us(&self) -> &[u64] {
        &self.per_token_decode_us
    }

    pub fn gpu_sample_count(&self) -> usize {
        self.gpu_samples.len()
    }

    pub fn gpu_timing_failure_count(&self) -> usize {
        self.gpu_timing_failures
    }

    pub fn nvtx_enabled(&self) -> bool {
        self.nvtx_enabled
    }

    pub fn nvtx_range_count(&self) -> usize {
        self.nvtx_range_count
    }

    pub fn by_section(&self) -> Vec<SectionRollup> {
        section_rollups(
            self.samples
                .iter()
                .map(|sample| (sample.section, sample.elapsed_us)),
        )
    }

    pub fn by_call_site(&self) -> Vec<CallSiteRollup> {
        call_site_rollups(
            self.samples
                .iter()
                .map(|sample| (sample.call_site.as_str(), sample.section, sample.elapsed_us)),
        )
    }

    pub fn by_gpu_section(&self) -> Vec<SectionRollup> {
        section_rollups(
            self.gpu_samples
                .iter()
                .map(|sample| (sample.section, sample.elapsed_us)),
        )
    }

    pub fn by_gpu_call_site(&self) -> Vec<CallSiteRollup> {
        call_site_rollups(
            self.gpu_samples
                .iter()
                .map(|sample| (sample.call_site.as_str(), sample.section, sample.elapsed_us)),
        )
    }
}

fn section_rollups<'a>(samples: impl IntoIterator<Item = (&'a str, u64)>) -> Vec<SectionRollup> {
    let mut total = 0u64;
    let mut groups: BTreeMap<&str, Vec<u64>> = BTreeMap::new();
    for (section, elapsed_us) in samples {
        total += elapsed_us;
        groups.entry(section).or_default().push(elapsed_us);
    }
    let mut rows: Vec<_> = groups
        .into_iter()
        .map(|(section, samples)| {
            let stats = sample_stats(samples);
            SectionRollup {
                section: section.to_string(),
                calls: stats.calls,
                total_us: stats.total_us,
                mean_us: stats.mean_us,
                min_us: stats.min_us,
                p50_us: stats.p50_us,
                p95_us: stats.p95_us,
                p99_us: stats.p99_us,
                max_us: stats.max_us,
                pct: pct(stats.total_us, total),
            }
        })
        .collect();
    rows.sort_by(|left, right| {
        right
            .total_us
            .cmp(&left.total_us)
            .then(left.section.cmp(&right.section))
    });
    rows
}

fn call_site_rollups<'a>(
    samples: impl IntoIterator<Item = (&'a str, &'a str, u64)>,
) -> Vec<CallSiteRollup> {
    let mut total = 0u64;
    let mut groups: BTreeMap<(&str, &str), Vec<u64>> = BTreeMap::new();
    for (call_site, section, elapsed_us) in samples {
        total += elapsed_us;
        groups
            .entry((call_site, section))
            .or_default()
            .push(elapsed_us);
    }
    let mut rows: Vec<_> = groups
        .into_iter()
        .map(|((call_site, section), samples)| {
            let stats = sample_stats(samples);
            CallSiteRollup {
                call_site: call_site.to_string(),
                section: section.to_string(),
                calls: stats.calls,
                total_us: stats.total_us,
                mean_us: stats.mean_us,
                min_us: stats.min_us,
                p50_us: stats.p50_us,
                p95_us: stats.p95_us,
                p99_us: stats.p99_us,
                max_us: stats.max_us,
                pct: pct(stats.total_us, total),
            }
        })
        .collect();
    rows.sort_by(|left, right| {
        right
            .total_us
            .cmp(&left.total_us)
            .then(left.call_site.cmp(&right.call_site))
            .then(left.section.cmp(&right.section))
    });
    rows
}

#[derive(Clone, Copy)]
struct SampleStats {
    calls: usize,
    total_us: u64,
    mean_us: f64,
    min_us: u64,
    p50_us: u64,
    p95_us: u64,
    p99_us: u64,
    max_us: u64,
}

fn sample_stats(mut samples: Vec<u64>) -> SampleStats {
    debug_assert!(!samples.is_empty());
    samples.sort_unstable();
    let total_us = samples.iter().sum();
    SampleStats {
        calls: samples.len(),
        total_us,
        mean_us: total_us as f64 / samples.len() as f64,
        min_us: samples[0],
        p50_us: percentile(&samples, 0.50),
        p95_us: percentile(&samples, 0.95),
        p99_us: percentile(&samples, 0.99),
        max_us: samples[samples.len() - 1],
    }
}

fn percentile(sorted: &[u64], quantile: f64) -> u64 {
    let idx = ((sorted.len() as f64 - 1.0) * quantile).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn pct(value: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        value as f64 / total as f64 * 100.0
    }
}

fn micros(duration: Duration) -> u64 {
    duration.as_micros().min(u64::MAX as u128) as u64
}

fn nvtx_enabled_from_env() -> bool {
    nvtx_enabled_value(env::var("PEGAINFER_DSV2_LITE_NVTX").ok().as_deref())
}

fn nvtx_enabled_value(value: Option<&str>) -> bool {
    matches!(
        value,
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}
