use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use clap::ValueEnum;
use pegainfer_higgs_audio::compare::OneStepSemanticTolerances;
use pegainfer_higgs_audio::compare::OneStepTolerances;
use pegainfer_higgs_audio::compare::compare_one_step_files;
use pegainfer_higgs_audio::compare::compare_one_step_semantic_files;
use pegainfer_higgs_audio::compare::ensure_comparison_passed;
use pegainfer_higgs_audio::compare::ensure_semantic_comparison_passed;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CompareMode {
    /// Enforce exact tensor identity plus tight numeric tolerances.
    Strict,
    /// Print strict drift diagnostics, then enforce semantic runtime parity.
    Semantic,
}

#[derive(Parser)]
#[command(about = "Compare a Higgs Audio one-step actual safetensors dump against the golden")]
struct Args {
    #[arg(long)]
    golden: PathBuf,
    #[arg(long)]
    actual: PathBuf,
    #[arg(long, value_enum, default_value_t = CompareMode::Strict)]
    mode: CompareMode,
    #[arg(long, default_value_t = OneStepTolerances::default().hidden_abs_tol)]
    hidden_abs_tol: f32,
    #[arg(long, default_value_t = OneStepTolerances::default().hidden_mean_abs_tol)]
    hidden_mean_abs_tol: f32,
    #[arg(long, default_value_t = OneStepTolerances::default().logits_abs_tol)]
    logits_abs_tol: f32,
    #[arg(long, default_value_t = OneStepTolerances::default().logits_mean_abs_tol)]
    logits_mean_abs_tol: f32,
    #[arg(long, default_value_t = OneStepTolerances::default().top_logprobs_abs_tol)]
    top_logprobs_abs_tol: f32,
    #[arg(long, default_value_t = OneStepTolerances::default().top_logprobs_mean_abs_tol)]
    top_logprobs_mean_abs_tol: f32,
    #[arg(long, default_value_t = OneStepSemanticTolerances::default().hidden_cosine_min)]
    hidden_cosine_min: f32,
    #[arg(long, default_value_t = OneStepSemanticTolerances::default().logits_cosine_min)]
    logits_cosine_min: f32,
    #[arg(long, default_value_t = OneStepSemanticTolerances::default().argmax_regret_tol)]
    argmax_regret_tol: f32,
    #[arg(long, default_value_t = OneStepSemanticTolerances::default().top64_min_overlap)]
    top64_min_overlap: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tolerances = OneStepTolerances {
        hidden_abs_tol: args.hidden_abs_tol,
        hidden_mean_abs_tol: args.hidden_mean_abs_tol,
        logits_abs_tol: args.logits_abs_tol,
        logits_mean_abs_tol: args.logits_mean_abs_tol,
        top_logprobs_abs_tol: args.top_logprobs_abs_tol,
        top_logprobs_mean_abs_tol: args.top_logprobs_mean_abs_tol,
    };
    let comparison = compare_one_step_files(&args.golden, &args.actual, tolerances)?;
    println!("higgs one-step strict comparison:");
    for tensor in &comparison.tensors {
        println!(
            "  {:32} pass={} elems={} exact_mismatch={} max_abs={:.6} mean_abs={:.6} rmse={:.6} p99_abs={:.6} abs_tol={:.6} mean_tol={:.6}",
            tensor.name,
            tensor.passed,
            tensor.elements,
            tensor.exact_mismatches,
            tensor.max_abs,
            tensor.mean_abs,
            tensor.rmse,
            tensor.p99_abs,
            tensor.abs_tol,
            tensor.mean_abs_tol
        );
    }

    match args.mode {
        CompareMode::Strict => {
            ensure_comparison_passed(&comparison)?;
            println!("higgs one-step strict comparison: ok");
        }
        CompareMode::Semantic => {
            println!(
                "higgs one-step strict comparison: passed={} diagnostic_only=true",
                comparison.passed()
            );
            let semantic_tolerances = OneStepSemanticTolerances {
                hidden_cosine_min: args.hidden_cosine_min,
                logits_cosine_min: args.logits_cosine_min,
                argmax_regret_tol: args.argmax_regret_tol,
                top64_min_overlap: args.top64_min_overlap,
            };
            let semantic =
                compare_one_step_semantic_files(&args.golden, &args.actual, semantic_tolerances)?;
            println!("higgs one-step semantic comparison:");
            println!(
                "  prompt_exact={} argmax_exact={} hidden_cosine={:.9} hidden_cosine_min={:.9}",
                semantic.prompt_exact,
                semantic.audio_argmax_exact,
                semantic.hidden_cosine,
                semantic.tolerances.hidden_cosine_min
            );
            println!(
                "  logits_cosine={:.9} logits_cosine_min={:.9} max_argmax_regret={:.6} argmax_regret_tol={:.6}",
                semantic.logits_cosine,
                semantic.tolerances.logits_cosine_min,
                semantic.max_argmax_regret,
                semantic.tolerances.argmax_regret_tol
            );
            println!(
                "  top64_min_overlap={} top64_mean_overlap={:.2} top64_min_overlap_tol={}",
                semantic.top64_min_overlap,
                semantic.top64_mean_overlap,
                semantic.tolerances.top64_min_overlap
            );
            ensure_semantic_comparison_passed(&semantic)?;
            println!("higgs one-step semantic comparison: ok");
        }
    }
    Ok(())
}
