use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use anyhow::bail;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Qwen3RuntimeSourcePath<'a> {
    BodyView(&'a Path),
    ConfigAlias(&'a Path),
    AutoConfigAlias(PathBuf),
}

pub fn select_qwen3_runtime_source<'a>(
    qwen3_body_dir: Option<&'a Path>,
    qwen3_config_dir: Option<&'a Path>,
    out: &Path,
) -> Result<Qwen3RuntimeSourcePath<'a>> {
    match (qwen3_body_dir, qwen3_config_dir) {
        (Some(qwen3_body_dir), None) => Ok(Qwen3RuntimeSourcePath::BodyView(qwen3_body_dir)),
        (None, Some(qwen3_config_dir)) => Ok(Qwen3RuntimeSourcePath::ConfigAlias(qwen3_config_dir)),
        (None, None) => Ok(Qwen3RuntimeSourcePath::AutoConfigAlias(
            default_qwen3_config_dir(out),
        )),
        (Some(_), Some(_)) => bail!("choose only one Qwen3 runtime source"),
    }
}

pub fn default_qwen3_config_dir(out: &Path) -> PathBuf {
    out.parent()
        .map(|parent| parent.join("higgs-qwen3-config-view"))
        .unwrap_or_else(|| PathBuf::from("higgs-qwen3-config-view"))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::Qwen3RuntimeSourcePath;
    use super::default_qwen3_config_dir;
    use super::select_qwen3_runtime_source;

    #[test]
    fn default_config_view_lives_next_to_actual_output() {
        assert_eq!(
            default_qwen3_config_dir(Path::new("/tmp/higgs/actual/out.safetensors")),
            Path::new("/tmp/higgs/actual/higgs-qwen3-config-view")
        );
    }

    #[test]
    fn default_config_view_falls_back_for_bare_output_name() {
        assert_eq!(
            default_qwen3_config_dir(Path::new("out.safetensors")),
            Path::new("higgs-qwen3-config-view")
        );
    }

    #[test]
    fn source_selection_prefers_explicit_body_view() {
        let body = Path::new("/tmp/qwen3-body-view");
        let selected =
            select_qwen3_runtime_source(Some(body), None, Path::new("/tmp/out.safetensors"))
                .unwrap();
        assert_eq!(selected, Qwen3RuntimeSourcePath::BodyView(body));
    }

    #[test]
    fn source_selection_prefers_explicit_config_alias() {
        let config = Path::new("/tmp/qwen3-config-view");
        let selected =
            select_qwen3_runtime_source(None, Some(config), Path::new("/tmp/out.safetensors"))
                .unwrap();
        assert_eq!(selected, Qwen3RuntimeSourcePath::ConfigAlias(config));
    }

    #[test]
    fn source_selection_defaults_to_auto_config_alias() {
        let selected =
            select_qwen3_runtime_source(None, None, Path::new("/tmp/higgs/actual/out.safetensors"))
                .unwrap();
        assert_eq!(
            selected,
            Qwen3RuntimeSourcePath::AutoConfigAlias(
                Path::new("/tmp/higgs/actual/higgs-qwen3-config-view").to_path_buf()
            )
        );
    }

    #[test]
    fn source_selection_rejects_ambiguous_runtime_source() {
        let error = select_qwen3_runtime_source(
            Some(Path::new("/tmp/body")),
            Some(Path::new("/tmp/config")),
            Path::new("/tmp/out.safetensors"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("choose only one"));
    }
}
