use std::collections::HashMap;
use std::ffi::CStr;
use std::ffi::c_void;
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
    pub json_path: PathBuf,
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
        attributes: KernelAttributes,
        /// `None` when the node carries no staged `kernelParams` array (e.g.
        /// packed `extra` launch buffers, which stream capture never produces).
        params: Option<Vec<KernelParamDump>>,
    },
    Other {
        node_type: String,
    },
}

/// Per-function facts a replayer needs beyond the launch configuration.
#[derive(Clone, Copy)]
struct KernelAttributes {
    num_regs: i32,
    static_shared_bytes: i32,
    const_bytes: i32,
    local_bytes: i32,
    ptx_version: i32,
    binary_version: i32,
}

/// One kernel parameter's staged bytes, captured at graph-capture time.
struct KernelParamDump {
    offset: usize,
    size: usize,
    bytes: Vec<u8>,
    /// Present when an 8-byte value resolves as a live CUDA allocation.
    /// Advisory: an integer that collides with an allocation also matches.
    pointer: Option<PointerRange>,
}

struct PointerRange {
    memory_type: &'static str,
    range_start: u64,
    range_size: usize,
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
    // 12.3 for cuFuncGetName, 12.4 for cuFuncGetParamInfo.
    const MIN_DRIVER_API_VERSION: i32 = 12_040;

    let mut version = 0i32;
    check(
        unsafe { sys::cuDriverGetVersion(&raw mut version) },
        "cuDriverGetVersion",
    )?;
    ensure!(
        version >= MIN_DRIVER_API_VERSION,
        "--dump-graph-png requires CUDA driver API 12.4 or newer for kernel names \
         and parameter layouts; found {}",
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
        let json_path = png_path.with_extension("json");
        std::fs::write(&json_path, graph.machine_json(title)?)
            .with_context(|| format!("write CUDA Graph JSON to {}", json_path.display()))?;
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
            json_path,
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
                        attributes,
                        params,
                    } => GraphNodeKind::Kernel {
                        raw_symbol,
                        demangled: demangled
                            .next()
                            .expect("one demangled name per kernel node"),
                        grid,
                        block,
                        dynamic_shared_mem_bytes,
                        attributes,
                        params,
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
        attributes: KernelAttributes,
        params: Option<Vec<KernelParamDump>>,
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
    let attributes = kernel_attributes(params.func)?;
    let staged_params = kernel_param_dumps(params.func, params.kernelParams)
        .with_context(|| format!("read staged kernel parameters of `{raw_symbol}`"))?;
    Ok(RawNodeKind::Kernel {
        raw_symbol,
        grid: [params.gridDimX, params.gridDimY, params.gridDimZ],
        block: [params.blockDimX, params.blockDimY, params.blockDimZ],
        dynamic_shared_mem_bytes: params.sharedMemBytes,
        attributes,
        params: staged_params,
    })
}

fn kernel_attributes(func: sys::CUfunction) -> Result<KernelAttributes> {
    use sys::CUfunction_attribute_enum as Attr;
    let attribute = |attribute: sys::CUfunction_attribute, what: &str| -> Result<i32> {
        let mut value = 0i32;
        check(
            unsafe { sys::cuFuncGetAttribute(&raw mut value, attribute, func) },
            what,
        )?;
        Ok(value)
    };
    Ok(KernelAttributes {
        num_regs: attribute(
            Attr::CU_FUNC_ATTRIBUTE_NUM_REGS,
            "cuFuncGetAttribute(num_regs)",
        )?,
        static_shared_bytes: attribute(
            Attr::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
            "cuFuncGetAttribute(shared_size_bytes)",
        )?,
        const_bytes: attribute(
            Attr::CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
            "cuFuncGetAttribute(const_size_bytes)",
        )?,
        local_bytes: attribute(
            Attr::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
            "cuFuncGetAttribute(local_size_bytes)",
        )?,
        ptx_version: attribute(
            Attr::CU_FUNC_ATTRIBUTE_PTX_VERSION,
            "cuFuncGetAttribute(ptx_version)",
        )?,
        binary_version: attribute(
            Attr::CU_FUNC_ATTRIBUTE_BINARY_VERSION,
            "cuFuncGetAttribute(binary_version)",
        )?,
    })
}

/// Read every staged parameter value of one captured kernel node.
///
/// `cuFuncGetParamInfo` walks the function's parameter layout; the graph node's
/// `kernelParams` array points at graph-owned host staging for each value.
fn kernel_param_dumps(
    func: sys::CUfunction,
    kernel_params: *mut *mut c_void,
) -> Result<Option<Vec<KernelParamDump>>> {
    // The driver caps a kernel's parameter buffer at a few KiB; a runaway index
    // here means the walk is broken, not that the kernel is huge.
    const MAX_PARAMS: usize = 4_096;
    if kernel_params.is_null() {
        return Ok(None);
    }
    let mut dumps = Vec::new();
    for index in 0..MAX_PARAMS {
        let mut offset = 0usize;
        let mut size = 0usize;
        let result =
            unsafe { sys::cuFuncGetParamInfo(func, index, &raw mut offset, &raw mut size) };
        if result == sys::CUresult::CUDA_ERROR_INVALID_VALUE {
            break;
        }
        check(result, "cuFuncGetParamInfo")?;
        let staged = unsafe { *kernel_params.add(index) };
        ensure!(
            !staged.is_null(),
            "kernel parameter {index} has no staged value"
        );
        let bytes = unsafe { std::slice::from_raw_parts(staged.cast::<u8>(), size) }.to_vec();
        let pointer = (size == 8)
            .then(|| {
                let value = u64::from_le_bytes(bytes[..8].try_into().expect("size == 8"));
                classify_pointer(value)
            })
            .flatten();
        dumps.push(KernelParamDump {
            offset,
            size,
            bytes,
            pointer,
        });
    }
    Ok(Some(dumps))
}

/// Resolve an 8-byte parameter value against the CUDA allocation map.
fn classify_pointer(value: u64) -> Option<PointerRange> {
    use sys::CUpointer_attribute_enum as Attr;
    if value == 0 {
        return None;
    }
    let mut memory_type = 0u32;
    let result = unsafe {
        sys::cuPointerGetAttribute(
            (&raw mut memory_type).cast::<c_void>(),
            Attr::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
            value,
        )
    };
    if result != sys::CUresult::CUDA_SUCCESS {
        return None;
    }
    let memory_type = match memory_type {
        1 => "host",
        2 => "device",
        3 => "array",
        4 => "unified",
        _ => "unknown",
    };
    let mut range_start = 0u64;
    let mut range_size = 0usize;
    // Best-effort: a pointer with a memory type but no queryable range keeps
    // the zero placeholders rather than failing the dump.
    let _ = unsafe {
        sys::cuPointerGetAttribute(
            (&raw mut range_start).cast::<c_void>(),
            Attr::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
            value,
        )
    };
    let _ = unsafe {
        sys::cuPointerGetAttribute(
            (&raw mut range_size).cast::<c_void>(),
            Attr::CU_POINTER_ATTRIBUTE_RANGE_SIZE,
            value,
        )
    };
    Some(PointerRange {
        memory_type,
        range_start,
        range_size,
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
    /// Machine-readable dump: everything the DOT carries plus per-kernel
    /// function attributes and staged parameter bytes, for downstream tooling
    /// (kernel ledger, capsule extraction) rather than human inspection.
    fn machine_json(&self, title: &str) -> Result<String> {
        let mut version = 0i32;
        check(
            unsafe { sys::cuDriverGetVersion(&raw mut version) },
            "cuDriverGetVersion",
        )?;
        let nodes = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| match &node.kind {
                GraphNodeKind::Kernel {
                    raw_symbol,
                    demangled,
                    grid,
                    block,
                    dynamic_shared_mem_bytes,
                    attributes,
                    params,
                } => serde_json::json!({
                    "id": index,
                    "type": "kernel",
                    "symbol": raw_symbol,
                    "name": demangled,
                    "grid": grid,
                    "block": block,
                    "dynamic_shared_mem_bytes": dynamic_shared_mem_bytes,
                    "attributes": {
                        "num_regs": attributes.num_regs,
                        "static_shared_bytes": attributes.static_shared_bytes,
                        "const_bytes": attributes.const_bytes,
                        "local_bytes": attributes.local_bytes,
                        "ptx_version": attributes.ptx_version,
                        "binary_version": attributes.binary_version,
                    },
                    "params": params.as_ref().map(|params| {
                        params
                            .iter()
                            .map(|param| {
                                serde_json::json!({
                                    "offset": param.offset,
                                    "size": param.size,
                                    "data": hex(&param.bytes),
                                    "pointer": param.pointer.as_ref().map(|pointer| {
                                        serde_json::json!({
                                            "memory_type": pointer.memory_type,
                                            "range_start": format!("{:#x}", pointer.range_start),
                                            "range_size": pointer.range_size,
                                        })
                                    }),
                                })
                            })
                            .collect::<Vec<_>>()
                    }),
                }),
                GraphNodeKind::Other { node_type } => serde_json::json!({
                    "id": index,
                    "type": node_type,
                }),
            })
            .collect::<Vec<_>>();
        let edges = self
            .edges
            .iter()
            .map(|edge| {
                serde_json::json!({
                    "from": edge.from,
                    "to": edge.to,
                    "from_port": edge.from_port,
                    "to_port": edge.to_port,
                    "dependency_type": dependency_type_name(edge.dependency_type),
                })
            })
            .collect::<Vec<_>>();
        let dump = serde_json::json!({
            "schema": "pegainfer-cuda-graph-dump/v1",
            "title": title,
            "driver_api_version": format_driver_api_version(version),
            "nodes": nodes,
            "edges": edges,
        });
        serde_json::to_string_pretty(&dump).context("serialize CUDA Graph JSON dump")
    }

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
                    ..
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

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
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
