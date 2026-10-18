use std::collections::HashMap;
use std::ffi::CStr;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::process::Output;
use std::process::Stdio;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::sys::CUgraphNode;
use cudarc::driver::sys::{self};

use super::CudaGraphState;
use super::check;

mod render;

#[derive(Clone, Debug)]
pub struct CudaGraphDumpSummary {
    pub nodes: usize,
    pub edges: usize,
    pub kernels: usize,
    pub dot_path: PathBuf,
    pub png_path: PathBuf,
}

struct GraphDescription {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
}

struct GraphNode {
    kind: GraphNodeKind,
}

#[derive(Clone, Copy)]
struct GraphEdge {
    from: usize,
    to: usize,
    from_port: u8,
    to_port: u8,
    dependency_type: u8,
}

impl GraphEdge {
    #[cfg(test)]
    fn ordinary(from: usize, to: usize) -> Self {
        Self {
            from,
            to,
            from_port: 0,
            to_port: 0,
            dependency_type: 0,
        }
    }
}

enum GraphNodeKind {
    Kernel {
        raw_symbol: String,
        demangled: String,
        grid: [u32; 3],
        block: [u32; 3],
        dynamic_shared_mem_bytes: u32,
    },
    Other {
        node_type: String,
    },
}

pub fn validate_graph_dump_request(png_path: &Path) -> Result<()> {
    ensure!(
        png_path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("png")),
        "--dump-graph-png expects a .png output path, got {}",
        png_path.display()
    );
    let parent = png_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).with_context(|| {
        format!(
            "create CUDA Graph dump output directory {}",
            parent.display()
        )
    })?;
    require_graph_dump_driver()?;
    require_tool("dot", &["-Tpng:cairo"], "Graphviz Cairo PNG renderer")?;
    require_tool("c++filt", &["--version"], "C++ demangler")?;
    Ok(())
}

fn require_graph_dump_driver() -> Result<()> {
    const MIN_DRIVER_API_VERSION: i32 = 12_030;

    let mut version = 0i32;
    check(
        unsafe { sys::cuDriverGetVersion(&raw mut version) },
        "cuDriverGetVersion",
    )?;
    ensure!(
        version >= MIN_DRIVER_API_VERSION,
        "--dump-graph-png requires CUDA driver API 12.3 or newer for kernel names; found {}",
        format_driver_api_version(version)
    );
    Ok(())
}

fn format_driver_api_version(version: i32) -> String {
    format!("{}.{}", version / 1000, version % 1000 / 10)
}

fn require_tool(program: &str, args: &[&str], description: &str) -> Result<()> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("{description} `{program}` is required for CUDA Graph export"))?;
    ensure!(
        output.status.success(),
        "{description} `{program}` check failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

impl CudaGraphState {
    pub fn dump_png(&self, png_path: &Path, title: &str) -> Result<CudaGraphDumpSummary> {
        ensure!(
            self.is_captured(),
            "CUDA graph dump requested before capture"
        );
        let graph = self.inspect()?;
        let dot_path = png_path.with_extension("dot");
        std::fs::write(&dot_path, graph.detailed_dot())
            .with_context(|| format!("write detailed CUDA Graph DOT to {}", dot_path.display()))?;
        render_png(&graph.human_dot(title), png_path)?;
        Ok(CudaGraphDumpSummary {
            nodes: graph.nodes.len(),
            edges: graph.edges.len(),
            kernels: graph
                .nodes
                .iter()
                .filter(|node| matches!(&node.kind, GraphNodeKind::Kernel { .. }))
                .count(),
            dot_path,
            png_path: png_path.to_path_buf(),
        })
    }

    fn inspect(&self) -> Result<GraphDescription> {
        let handles = graph_nodes(self.graph)?;
        let handle_to_index: HashMap<usize, usize> = handles
            .iter()
            .enumerate()
            .map(|(index, &handle)| (handle as usize, index))
            .collect();
        let raw_kinds = handles
            .iter()
            .map(|&handle| inspect_node(handle))
            .collect::<Result<Vec<_>>>()?;
        let symbols = raw_kinds
            .iter()
            .filter_map(|kind| match kind {
                RawNodeKind::Kernel { raw_symbol, .. } => Some(raw_symbol.as_str()),
                RawNodeKind::Other { .. } => None,
            })
            .collect::<Vec<_>>();
        let demangled = demangle(&symbols)?;
        let mut demangled = demangled.into_iter();
        let nodes = raw_kinds
            .into_iter()
            .map(|kind| GraphNode {
                kind: match kind {
                    RawNodeKind::Kernel {
                        raw_symbol,
                        grid,
                        block,
                        dynamic_shared_mem_bytes,
                    } => GraphNodeKind::Kernel {
                        raw_symbol,
                        demangled: demangled
                            .next()
                            .expect("one demangled name per kernel node"),
                        grid,
                        block,
                        dynamic_shared_mem_bytes,
                    },
                    RawNodeKind::Other { node_type } => GraphNodeKind::Other { node_type },
                },
            })
            .collect();
        let edges = graph_edges(self.graph)?
            .into_iter()
            .map(|(from, to, data)| {
                let from = handle_to_index
                    .get(&(from as usize))
                    .copied()
                    .context("CUDA graph edge source is absent from the node list")?;
                let to = handle_to_index
                    .get(&(to as usize))
                    .copied()
                    .context("CUDA graph edge destination is absent from the node list")?;
                Ok(GraphEdge {
                    from,
                    to,
                    from_port: data.from_port,
                    to_port: data.to_port,
                    dependency_type: data.type_,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(GraphDescription { nodes, edges })
    }
}

enum RawNodeKind {
    Kernel {
        raw_symbol: String,
        grid: [u32; 3],
        block: [u32; 3],
        dynamic_shared_mem_bytes: u32,
    },
    Other {
        node_type: String,
    },
}

fn graph_nodes(graph: sys::CUgraph) -> Result<Vec<CUgraphNode>> {
    let mut count = 0usize;
    check(
        unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &raw mut count) },
        "cuGraphGetNodes(count)",
    )?;
    let mut nodes = vec![std::ptr::null_mut(); count];
    check(
        unsafe { sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &raw mut count) },
        "cuGraphGetNodes(nodes)",
    )?;
    nodes.truncate(count);
    Ok(nodes)
}

fn graph_edges(
    graph: sys::CUgraph,
) -> Result<Vec<(CUgraphNode, CUgraphNode, sys::CUgraphEdgeData)>> {
    let mut count = 0usize;
    check(
        unsafe {
            sys::cuGraphGetEdges_v2(
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &raw mut count,
            )
        },
        "cuGraphGetEdges(count)",
    )?;
    let mut from = vec![std::ptr::null_mut(); count];
    let mut to = vec![std::ptr::null_mut(); count];
    let mut data = vec![
        sys::CUgraphEdgeData {
            from_port: 0,
            to_port: 0,
            type_: 0,
            reserved: [0; 5],
        };
        count
    ];
    check(
        unsafe {
            sys::cuGraphGetEdges_v2(
                graph,
                from.as_mut_ptr(),
                to.as_mut_ptr(),
                data.as_mut_ptr(),
                &raw mut count,
            )
        },
        "cuGraphGetEdges(edges)",
    )?;
    Ok(from
        .into_iter()
        .zip(to)
        .zip(data)
        .take(count)
        .map(|((from, to), data)| (from, to, data))
        .collect())
}

fn inspect_node(node: CUgraphNode) -> Result<RawNodeKind> {
    let mut node_type = std::mem::MaybeUninit::uninit();
    check(
        unsafe { sys::cuGraphNodeGetType(node, node_type.as_mut_ptr()) },
        "cuGraphNodeGetType",
    )?;
    let node_type = unsafe { node_type.assume_init() };
    if node_type != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
        return Ok(RawNodeKind::Other {
            node_type: format!("{node_type:?}")
                .trim_start_matches("CU_GRAPH_NODE_TYPE_")
                .to_ascii_lowercase(),
        });
    }

    let mut params = std::mem::MaybeUninit::<sys::CUDA_KERNEL_NODE_PARAMS>::zeroed();
    check(
        unsafe { sys::cuGraphKernelNodeGetParams_v2(node, params.as_mut_ptr()) },
        "cuGraphKernelNodeGetParams",
    )?;
    let params = unsafe { params.assume_init() };
    ensure!(
        !params.func.is_null(),
        "CUDA graph kernel node has no CUfunction"
    );
    let mut name = std::ptr::null();
    check(
        unsafe { sys::cuFuncGetName(&raw mut name, params.func) },
        "cuFuncGetName",
    )?;
    ensure!(!name.is_null(), "cuFuncGetName returned a null name");
    let raw_symbol = unsafe { CStr::from_ptr(name) }
        .to_str()
        .context("CUDA kernel name is not UTF-8")?
        .to_owned();
    Ok(RawNodeKind::Kernel {
        raw_symbol,
        grid: [params.gridDimX, params.gridDimY, params.gridDimZ],
        block: [params.blockDimX, params.blockDimY, params.blockDimZ],
        dynamic_shared_mem_bytes: params.sharedMemBytes,
    })
}

fn demangle(symbols: &[&str]) -> Result<Vec<String>> {
    if symbols.is_empty() {
        return Ok(Vec::new());
    }
    let child = Command::new("c++filt")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn C++ demangler `c++filt`")?;
    let mut input = String::new();
    for symbol in symbols {
        writeln!(input, "{symbol}").expect("writing to a String cannot fail");
    }
    let output = communicate(child, input, "c++filt")?;
    ensure!(
        output.status.success(),
        "c++filt failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let names = String::from_utf8(output.stdout)
        .context("c++filt output is not UTF-8")?
        .lines()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    ensure!(
        names.len() == symbols.len(),
        "c++filt returned {} names for {} symbols",
        names.len(),
        symbols.len()
    );
    Ok(names)
}

fn render_png(dot: &str, png_path: &Path) -> Result<()> {
    let child = Command::new("dot")
        .args(["-Tpng:cairo", "-Gdpi=192", "-o"])
        .arg(png_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn Graphviz `dot`")?;
    let output = communicate(child, dot.to_owned(), "Graphviz")?;
    ensure!(
        output.status.success(),
        "Graphviz failed to render {}: {}",
        png_path.display(),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

fn communicate(mut child: Child, input: String, program: &'static str) -> Result<Output> {
    let mut stdin = child
        .stdin
        .take()
        .with_context(|| format!("open {program} stdin"))?;
    // Whole-step graphs are larger than an OS pipe. Feed stdin concurrently
    // while `wait_with_output` drains stdout/stderr, so neither side can fill
    // a pipe while waiting for the other side to make progress.
    let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes()));
    let output = child
        .wait_with_output()
        .with_context(|| format!("wait for {program}"))?;
    let write_result = writer
        .join()
        .map_err(|_| anyhow::anyhow!("{program} stdin writer panicked"))?;
    if output.status.success() {
        write_result.with_context(|| format!("write input to {program}"))?;
    }
    Ok(output)
}

impl GraphDescription {
    fn detailed_dot(&self) -> String {
        let mut dot = String::from("digraph cuda_graph_detailed {\n");
        dot.push_str("  graph [rankdir=TB];\n  node [shape=box];\n");
        for (index, node) in self.nodes.iter().enumerate() {
            let label = match &node.kind {
                GraphNodeKind::Kernel {
                    raw_symbol,
                    demangled,
                    grid,
                    block,
                    dynamic_shared_mem_bytes,
                } => format!(
                    "id={index}\\ntype=kernel\\nname={}\\nraw_symbol={}\\ngrid={}\\nblock={}\\ndynamic_shared_mem_bytes={dynamic_shared_mem_bytes}",
                    dot_escape(demangled),
                    dot_escape(raw_symbol),
                    dims(*grid),
                    dims(*block),
                ),
                GraphNodeKind::Other { node_type } => {
                    format!("id={index}\\ntype={}", dot_escape(node_type))
                }
            };
            let _ = writeln!(dot, "  n{index} [label=\"{label}\"];");
        }
        for edge in &self.edges {
            let _ = writeln!(
                dot,
                "  n{} -> n{} [label=\"from_port={}\\nto_port={}\\ndependency_type={}\"];",
                edge.from,
                edge.to,
                edge.from_port,
                edge.to_port,
                dependency_type_name(edge.dependency_type),
            );
        }
        dot.push_str("}\n");
        dot
    }
}

fn dims(dims: [u32; 3]) -> String {
    format!("({},{},{})", dims[0], dims[1], dims[2])
}

fn dependency_type_name(dependency_type: u8) -> String {
    match dependency_type {
        0 => "default".to_owned(),
        1 => "programmatic".to_owned(),
        other => format!("unknown({other})"),
    }
}

fn dot_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "")
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::GraphDescription;
    use super::GraphEdge;
    use super::GraphNode;
    use super::GraphNodeKind;
    use super::communicate;
    use super::demangle;
    use super::dot_escape;
    use super::format_driver_api_version;

    #[test]
    fn dot_label_escaping_preserves_graph_syntax() {
        assert_eq!(dot_escape("a\\b\n\"c\""), "a\\\\b\\n\\\"c\\\"");
    }

    #[test]
    fn formats_cuda_driver_api_version() {
        assert_eq!(format_driver_api_version(12_020), "12.2");
        assert_eq!(format_driver_api_version(12_030), "12.3");
        assert_eq!(format_driver_api_version(13_000), "13.0");
    }

    #[test]
    fn demangle_drains_output_larger_than_a_pipe() {
        if std::process::Command::new("c++filt")
            .arg("--version")
            .output()
            .is_err()
        {
            return;
        }
        let symbols = (0..2_048)
            .map(|index| format!("plain_symbol_{index:04}_{}", "x".repeat(96)))
            .collect::<Vec<_>>();
        let refs = symbols.iter().map(String::as_str).collect::<Vec<_>>();

        let names = demangle(&refs).expect("large c++filt stream");

        assert_eq!(names, symbols);
    }

    #[test]
    fn subprocess_communication_drains_stdout_and_stderr() {
        let child = std::process::Command::new("sh")
            .args([
                "-c",
                "while IFS= read -r line; do printf '%s\\n' \"$line\"; printf '%s\\n' \"$line\" >&2; done",
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn pipe stress child");
        let mut input = String::new();
        for index in 0..2_048 {
            let _ = writeln!(input, "line_{index:04}_{}", "x".repeat(96));
        }

        let output = communicate(child, input.clone(), "pipe stress child")
            .expect("communicate with pipe stress child");

        assert_eq!(output.stdout, input.as_bytes());
        assert_eq!(output.stderr, input.as_bytes());
    }

    #[test]
    fn dot_preserves_programmatic_edge_metadata() {
        let node = || GraphNode {
            kind: GraphNodeKind::Other {
                node_type: "empty".to_owned(),
            },
        };
        let graph = GraphDescription {
            nodes: vec![node(), node()],
            edges: vec![GraphEdge {
                from: 0,
                to: 1,
                from_port: 1,
                to_port: 0,
                dependency_type: 1,
            }],
        };

        let detailed = graph.detailed_dot();
        let human = graph.human_dot("programmatic edge");

        assert!(detailed.contains("from_port=1\\nto_port=0\\ndependency_type=programmatic"));
        assert!(human.contains("style=dashed, label=\"programmatic · port 1→0\""));
    }
}
