use std::collections::HashSet;
use std::fmt::Write as _;

use super::GraphDescription;
use super::GraphNode;
use super::GraphNodeKind;
use super::dependency_type_name;
use super::dims;
use super::dot_escape;

const MAX_VISIBLE_REPEAT_WIDTH: usize = 32;

#[derive(Clone, Copy)]
struct RepeatedRun {
    start: usize,
    width: usize,
    repetitions: usize,
}

impl RepeatedRun {
    fn end(self) -> usize {
        self.start + self.width * self.repetitions
    }

    fn display_cost(self) -> usize {
        if self.width <= MAX_VISIBLE_REPEAT_WIDTH {
            self.width
        } else {
            1
        }
    }

    fn saved_nodes(self) -> usize {
        self.width * self.repetitions - self.display_cost()
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum HumanNodeId {
    Original(usize),
    RepeatedSummary(usize),
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct HumanEdge {
    from: HumanNodeId,
    to: HumanNodeId,
    from_port: u8,
    to_port: u8,
    dependency_type: u8,
    after_repetitions: Option<usize>,
}

impl GraphDescription {
    pub(super) fn human_dot(&self, title: &str) -> String {
        let mut indegree = vec![0usize; self.nodes.len()];
        let mut outdegree = vec![0usize; self.nodes.len()];
        for edge in &self.edges {
            outdegree[edge.from] += 1;
            indegree[edge.to] += 1;
        }

        let mut dot = String::from("digraph cuda_graph {\n");
        let _ = writeln!(
            dot,
            "  graph [rankdir=TB, bgcolor=\"white\", pad=0.2, nodesep=0.25, ranksep=0.35, fontname=\"Helvetica\", label=\"{}\", labelloc=t, fontsize=18];",
            dot_escape(title)
        );
        dot.push_str(
            "  node [shape=box, style=\"rounded,filled\", color=\"#374151\", \
             fontname=\"Helvetica\", fontsize=10, margin=\"0.12,0.08\"];\n",
        );
        dot.push_str("  edge [color=\"#6B7280\", arrowsize=0.6];\n");
        let repeated = self.repeated_dag_runs();
        let mut cursor = 0usize;
        for (run_index, &run) in repeated.iter().enumerate() {
            for index in cursor..run.start {
                self.write_human_node(&mut dot, index, indegree[index], outdegree[index]);
            }
            self.write_repeated_run(&mut dot, run_index, run, &indegree, &outdegree);
            cursor = run.end();
        }
        for index in cursor..self.nodes.len() {
            self.write_human_node(&mut dot, index, indegree[index], outdegree[index]);
        }

        let mut written = HashSet::new();
        for edge in &self.edges {
            let from_run = run_membership(edge.from, &repeated);
            let to_run = run_membership(edge.to, &repeated);
            if matches!((from_run, to_run), (Some((left, left_copy, _)), Some((right, right_copy, _))) if left == right && left_copy != right_copy)
            {
                continue;
            }
            let projected = HumanEdge {
                from: project_node(edge.from, from_run, &repeated),
                to: project_node(edge.to, to_run, &repeated),
                from_port: edge.from_port,
                to_port: edge.to_port,
                dependency_type: edge.dependency_type,
                after_repetitions: from_run.and_then(|(run_index, copy, _)| {
                    let run = repeated[run_index];
                    (copy + 1 == run.repetitions
                        && to_run.is_none_or(|(to_run_index, _, _)| to_run_index != run_index))
                    .then_some(run.repetitions)
                }),
            };
            if projected.from != projected.to && written.insert(projected) {
                write_human_edge(&mut dot, projected);
            }
        }
        dot.push_str("}\n");
        dot
    }

    fn write_human_node(&self, dot: &mut String, index: usize, indegree: usize, outdegree: usize) {
        let fill = node_color(indegree, outdegree);
        let label = match &self.nodes[index].kind {
            GraphNodeKind::Kernel {
                demangled,
                grid,
                block,
                dynamic_shared_mem_bytes,
                ..
            } => format!(
                "{}\\ngrid={} block={} dynamic_smem={}",
                dot_escape(&compact_kernel_name(demangled)),
                dims(*grid),
                dims(*block),
                dynamic_shared_mem_bytes,
            ),
            GraphNodeKind::Other { node_type } => dot_escape(&node_type.to_ascii_uppercase()),
        };
        let _ = writeln!(dot, "  n{index} [fillcolor=\"{fill}\", label=\"{label}\"];");
    }

    fn write_repeated_run(
        &self,
        dot: &mut String,
        run_index: usize,
        run: RepeatedRun,
        indegree: &[usize],
        outdegree: &[usize],
    ) {
        let kernels = self.nodes[run.start..run.start + run.width]
            .iter()
            .filter(|node| matches!(&node.kind, GraphNodeKind::Kernel { .. }))
            .count();
        let instance_description = if kernels == run.width {
            format!("{} kernels/instance", run.width)
        } else {
            format!("{} nodes/instance · {kernels} kernels/instance", run.width)
        };
        let _ = writeln!(
            dot,
            "  subgraph cluster_repeated_{run_index} {{\n    label=\"Repeated physical block ×{} · {instance_description}\";\n    color=\"#9CA3AF\";\n    style=\"rounded,dashed\";",
            run.repetitions,
        );
        if run.width <= MAX_VISIBLE_REPEAT_WIDTH {
            for index in run.start..run.start + run.width {
                self.write_human_node(dot, index, indegree[index], outdegree[index]);
            }
        } else {
            let _ = writeln!(
                dot,
                "    r{run_index} [shape=component, fillcolor=\"#F3F4F6\", label=\"Folded repeated subgraph\\n{} nodes and {kernels} kernels per instance\\nphysical repetitions ×{}\"];",
                run.width, run.repetitions,
            );
        }
        dot.push_str("  }\n");
    }

    fn repeated_dag_runs(&self) -> Vec<RepeatedRun> {
        let signatures = self
            .nodes
            .iter()
            .map(GraphNode::signature)
            .collect::<Vec<_>>();
        let mut candidates = Vec::new();
        for start in 0..signatures.len() {
            let max_width = (signatures.len() - start) / 3;
            let mut primitive_widths = Vec::new();
            for width in 2..=max_width {
                let first = &signatures[start..start + width];
                if signatures[start + width..start + 2 * width] != *first
                    || signatures[start + 2 * width..start + 3 * width] != *first
                {
                    continue;
                }
                let mut repetitions = 3usize;
                while start + (repetitions + 1) * width <= signatures.len()
                    && signatures[start + repetitions * width..start + (repetitions + 1) * width]
                        == *first
                {
                    repetitions += 1;
                }
                let mut candidate = None;
                for repetitions in (3..=repetitions).rev() {
                    let run = RepeatedRun {
                        start,
                        width,
                        repetitions,
                    };
                    if self.is_repeated_dag(run) {
                        candidate = Some(run);
                        break;
                    }
                }
                let Some(candidate) = candidate else {
                    continue;
                };
                let bundled_primitive = primitive_widths.iter().copied().any(|primitive_width| {
                    width % primitive_width == 0
                        && first
                            .chunks_exact(primitive_width)
                            .all(|chunk| chunk == &first[..primitive_width])
                });
                if bundled_primitive {
                    continue;
                }
                primitive_widths.push(width);
                candidates.push(candidate);
            }
        }
        let runs = select_non_overlapping_runs(candidates);
        let saved = runs.iter().map(|run| run.saved_nodes()).sum::<usize>();
        if saved.saturating_mul(2) >= self.nodes.len() {
            runs
        } else {
            Vec::new()
        }
    }

    fn is_repeated_dag(&self, run: RepeatedRun) -> bool {
        let mut internal = vec![Vec::new(); run.repetitions];
        let mut boundaries = vec![Vec::new(); run.repetitions - 1];
        for edge in &self.edges {
            let from = repeated_node_position(edge.from, run);
            let to = repeated_node_position(edge.to, run);
            match (from, to) {
                (Some((from_copy, from_offset)), Some((to_copy, to_offset)))
                    if from_copy == to_copy =>
                {
                    internal[from_copy].push((
                        from_offset,
                        to_offset,
                        edge.from_port,
                        edge.to_port,
                        edge.dependency_type,
                    ));
                }
                (Some((from_copy, from_offset)), Some((to_copy, to_offset)))
                    if from_copy + 1 == to_copy =>
                {
                    boundaries[from_copy].push((
                        from_offset,
                        to_offset,
                        edge.from_port,
                        edge.to_port,
                        edge.dependency_type,
                    ));
                }
                (Some(_), Some(_)) => return false,
                (None, Some((to_copy, _))) if to_copy != 0 => return false,
                (Some((from_copy, _)), None) if from_copy + 1 != run.repetitions => return false,
                _ => {}
            }
        }
        for edges in internal.iter_mut().chain(boundaries.iter_mut()) {
            edges.sort_unstable();
        }
        boundaries.iter().all(|edges| !edges.is_empty())
            && internal.windows(2).all(|pair| pair[0] == pair[1])
            && boundaries.windows(2).all(|pair| pair[0] == pair[1])
    }
}

impl GraphNode {
    fn signature(&self) -> String {
        match &self.kind {
            GraphNodeKind::Kernel {
                raw_symbol,
                grid,
                block,
                dynamic_shared_mem_bytes,
                ..
            } => format!(
                "kernel|{raw_symbol}|{}|{}|{dynamic_shared_mem_bytes}",
                dims(*grid),
                dims(*block)
            ),
            GraphNodeKind::Other { node_type } => format!("other|{node_type}"),
        }
    }
}

fn repeated_node_position(index: usize, run: RepeatedRun) -> Option<(usize, usize)> {
    (run.start..run.end()).contains(&index).then(|| {
        let offset = index - run.start;
        (offset / run.width, offset % run.width)
    })
}

fn select_non_overlapping_runs(mut candidates: Vec<RepeatedRun>) -> Vec<RepeatedRun> {
    candidates.sort_unstable_by_key(|run| (run.end(), run.start, run.width, run.repetitions));
    let mut best_savings = vec![0usize; candidates.len() + 1];
    let mut take = vec![false; candidates.len()];
    let mut compatible_count = vec![0usize; candidates.len()];
    for (index, &candidate) in candidates.iter().enumerate() {
        let compatible = candidates[..index].partition_point(|run| run.end() <= candidate.start);
        compatible_count[index] = compatible;
        let with_candidate = best_savings[compatible] + candidate.saved_nodes();
        if with_candidate > best_savings[index] {
            best_savings[index + 1] = with_candidate;
            take[index] = true;
        } else {
            best_savings[index + 1] = best_savings[index];
        }
    }

    let mut selected = Vec::new();
    let mut cursor = candidates.len();
    while cursor > 0 {
        let index = cursor - 1;
        if take[index] {
            selected.push(candidates[index]);
            cursor = compatible_count[index];
        } else {
            cursor -= 1;
        }
    }
    selected.sort_unstable_by_key(|run| run.start);
    selected
}

fn run_membership(index: usize, runs: &[RepeatedRun]) -> Option<(usize, usize, usize)> {
    runs.iter().enumerate().find_map(|(run_index, &run)| {
        repeated_node_position(index, run).map(|(copy, offset)| (run_index, copy, offset))
    })
}

fn project_node(
    index: usize,
    membership: Option<(usize, usize, usize)>,
    runs: &[RepeatedRun],
) -> HumanNodeId {
    let Some((run_index, _, offset)) = membership else {
        return HumanNodeId::Original(index);
    };
    let run = runs[run_index];
    if run.width <= MAX_VISIBLE_REPEAT_WIDTH {
        HumanNodeId::Original(run.start + offset)
    } else {
        HumanNodeId::RepeatedSummary(run_index)
    }
}

fn write_human_edge(dot: &mut String, edge: HumanEdge) {
    dot.push_str("  ");
    write_human_node_id(dot, edge.from);
    dot.push_str(" -> ");
    write_human_node_id(dot, edge.to);
    let special = edge.from_port != 0 || edge.to_port != 0 || edge.dependency_type != 0;
    match (edge.after_repetitions, special) {
        (None, false) => dot.push_str(";\n"),
        (Some(repetitions), false) => {
            let _ = writeln!(dot, " [label=\"after ×{repetitions}\", fontsize=9];");
        }
        (after, true) => {
            let prefix = after
                .map(|repetitions| format!("after ×{repetitions} · "))
                .unwrap_or_default();
            let _ = writeln!(
                dot,
                " [color=\"#2563EB\", style=dashed, label=\"{prefix}{} · port {}→{}\", fontsize=8];",
                dependency_type_name(edge.dependency_type),
                edge.from_port,
                edge.to_port,
            );
        }
    }
}

fn write_human_node_id(dot: &mut String, node: HumanNodeId) {
    match node {
        HumanNodeId::Original(index) => {
            let _ = write!(dot, "n{index}");
        }
        HumanNodeId::RepeatedSummary(index) => {
            let _ = write!(dot, "r{index}");
        }
    }
}

fn node_color(indegree: usize, outdegree: usize) -> &'static str {
    if indegree == 0 {
        "#DCEEFF"
    } else if outdegree == 0 {
        "#E8DEFF"
    } else if indegree > 1 {
        "#FFE8C2"
    } else if outdegree > 1 {
        "#E3F6E8"
    } else {
        "#F5F5F5"
    }
}

fn compact_kernel_name(name: &str) -> String {
    const MAX_CHARS: usize = 72;

    if name.contains("internal::gemvx::kernel") {
        return "cuBLAS GEMV".to_owned();
    }
    let signature = name.rsplit_once('(').map_or(name, |(name, _)| name);
    let mut compact = String::with_capacity(signature.len());
    let mut template_depth = 0usize;
    for ch in signature.chars() {
        match ch {
            '<' => {
                if template_depth == 0 {
                    compact.push_str("<…>");
                }
                template_depth += 1;
            }
            '>' if template_depth > 0 => template_depth -= 1,
            _ if template_depth == 0 => compact.push(ch),
            _ => {}
        }
    }
    let leaf = compact.rsplit("::").next().unwrap_or(&compact);
    if leaf.chars().count() <= MAX_CHARS {
        return leaf.to_owned();
    }
    let prefix = leaf.chars().take(MAX_CHARS - 1).collect::<String>();
    format!("{prefix}…")
}

#[cfg(test)]
mod tests {
    use super::super::GraphDescription;
    use super::super::GraphEdge;
    use super::super::GraphNode;
    use super::super::GraphNodeKind;
    use super::compact_kernel_name;

    fn kernel(name: &str) -> GraphNode {
        GraphNode {
            kind: GraphNodeKind::Kernel {
                raw_symbol: name.to_owned(),
                demangled: format!("{name}()"),
                grid: [1, 1, 1],
                block: [32, 1, 1],
                dynamic_shared_mem_bytes: 0,
            },
        }
    }

    #[test]
    fn compact_name_drops_namespace_signature_and_template_body() {
        assert_eq!(
            compact_kernel_name("flashinfer::BatchDecodeKernel<128, foo::Bar>(float*, int)"),
            "BatchDecodeKernel<…>"
        );
        assert_eq!(
            compact_kernel_name(
                "std::enable_if<true, void>::type internal::gemvx::kernel<int>(int)"
            ),
            "cuBLAS GEMV"
        );
    }

    #[test]
    fn human_dot_folds_a_repeated_linear_block() {
        let names = ["head", "a", "b", "a", "b", "a", "b", "tail"];
        let graph = GraphDescription {
            nodes: names.into_iter().map(kernel).collect(),
            edges: (0..7)
                .map(|index| GraphEdge::ordinary(index, index + 1))
                .collect(),
        };

        let dot = graph.human_dot("test graph");

        assert!(dot.contains("label=\"test graph\""));
        assert!(dot.contains("Repeated physical block ×3 · 2 kernels/instance"));
        assert!(dot.contains("n0 -> n1"));
        assert!(dot.contains("n2 -> n7 [label=\"after ×3\""));
        assert!(!dot.contains("n3 [fillcolor"));
    }

    #[test]
    fn human_dot_folds_a_repeated_branched_dag() {
        let names = [
            "head", "fork", "left", "right", "join", "fork", "left", "right", "join", "fork",
            "left", "right", "join", "tail",
        ];
        let mut edges = vec![GraphEdge::ordinary(0, 1)];
        for start in [1, 5, 9] {
            edges.extend([
                GraphEdge::ordinary(start, start + 1),
                GraphEdge::ordinary(start, start + 2),
                GraphEdge::ordinary(start + 1, start + 3),
                GraphEdge::ordinary(start + 2, start + 3),
            ]);
        }
        edges.extend([
            GraphEdge::ordinary(4, 5),
            GraphEdge::ordinary(8, 9),
            GraphEdge::ordinary(12, 13),
        ]);
        let graph = GraphDescription {
            nodes: names.into_iter().map(kernel).collect(),
            edges,
        };

        let dot = graph.human_dot("branched graph");

        assert!(dot.contains("Repeated physical block ×3 · 4 kernels/instance"));
        assert!(dot.contains("n1 -> n2;"));
        assert!(dot.contains("n1 -> n3;"));
        assert!(dot.contains("n2 -> n4;"));
        assert!(dot.contains("n3 -> n4;"));
        assert!(dot.contains("n4 -> n13 [label=\"after ×3\""));
        assert!(!dot.contains("n5 [fillcolor"));
    }

    #[test]
    fn repeated_signatures_with_different_edges_are_not_folded() {
        let names = ["head", "a", "b", "a", "b", "a", "b", "tail"];
        let graph = GraphDescription {
            nodes: names.into_iter().map(kernel).collect(),
            edges: vec![
                GraphEdge::ordinary(0, 1),
                GraphEdge::ordinary(1, 2),
                GraphEdge::ordinary(2, 3),
                GraphEdge::ordinary(3, 5),
                GraphEdge::ordinary(5, 6),
                GraphEdge::ordinary(6, 7),
            ],
        };

        assert!(graph.repeated_dag_runs().is_empty());
    }

    #[test]
    fn disconnected_parallel_copies_are_not_folded_as_a_sequence() {
        let names = ["head", "a", "b", "a", "b", "a", "b", "tail"];
        let graph = GraphDescription {
            nodes: names.into_iter().map(kernel).collect(),
            edges: vec![
                GraphEdge::ordinary(0, 1),
                GraphEdge::ordinary(1, 2),
                GraphEdge::ordinary(3, 4),
                GraphEdge::ordinary(5, 6),
                GraphEdge::ordinary(6, 7),
            ],
        };

        assert!(graph.repeated_dag_runs().is_empty());
    }

    #[test]
    fn qwen_shape_uses_the_primitive_fourteen_kernel_period() {
        let mut names = vec!["head".to_owned()];
        for _ in 0..36 {
            names.extend((0..14).map(|index| format!("layer_kernel_{index}")));
        }
        names.push("tail".to_owned());
        let graph = GraphDescription {
            nodes: names.iter().map(|name| kernel(name)).collect(),
            edges: (0..names.len() - 1)
                .map(|index| GraphEdge::ordinary(index, index + 1))
                .collect(),
        };

        let dot = graph.human_dot("qwen graph");

        assert!(dot.contains("Repeated physical block ×36 · 14 kernels/instance"));
        assert!(!dot.contains("Folded repeated subgraph"));
    }

    #[test]
    fn large_repeated_dag_uses_one_summary_node() {
        let mut names = vec!["head".to_owned()];
        for _ in 0..3 {
            names.extend((0..33).map(|index| format!("layer_{index}")));
        }
        names.push("tail".to_owned());
        let graph = GraphDescription {
            nodes: names.iter().map(|name| kernel(name)).collect(),
            edges: (0..names.len() - 1)
                .map(|index| GraphEdge::ordinary(index, index + 1))
                .collect(),
        };

        let dot = graph.human_dot("large graph");

        assert!(dot.contains("Folded repeated subgraph\\n33 nodes and 33 kernels per instance"));
        assert!(dot.contains("n0 -> r0;"));
        assert!(dot.contains("r0 -> n100 [label=\"after ×3\""));
        assert!(!dot.contains("n1 [fillcolor"));
    }
}
