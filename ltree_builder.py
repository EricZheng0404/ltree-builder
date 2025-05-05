# ltree_builder.py
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

###############################################################################
# 1. Utilities
###############################################################################
@dataclass
class NodeStat:
    latencies: list = field(default_factory=list)

    def update(self, rt):
        # rt in CASPER is request time in ms (can be -1 for async or errors)
        if rt >= 0:
            self.latencies.append(rt)

    @property
    def p50(self):
        return np.percentile(self.latencies, 50) if self.latencies else None

    @property
    def minimum(self):
        return min(self.latencies) if self.latencies else None

    @property
    def maximum(self):
        return max(self.latencies) if self.latencies else None


###############################################################################
# 2. Per-trace tree (a thin wrapper around nested dicts)
###############################################################################
class TraceTree:
    """Builds a tree for a single trace using rpcid hierarchy (0.1.1.2 …)."""

    def __init__(self, df_trace):
        self.root = {}
        self.stats = defaultdict(NodeStat)
        self._build(df_trace)

    def _insert(self, path, rt):
        node = self.root
        for hop in path:
            node = node.setdefault(hop, {})
        self.stats[tuple(path)].update(rt)

    def _build(self, df_trace):
        # rpcid "0.1.2" → hops = [0,1,2]
        for _, row in df_trace.sort_values("rpcid").iterrows():
            rpcid = row["rpcid"]
            if all(part.isdigit() for part in rpcid.split(".")):
                hops = [int(h) for h in row["rpcid"].split(".")]
                self._insert(hops, row["rt"])

###############################################################################
# 3. L-tree merger
###############################################################################
class LTreemerger:
    """Merge many TraceTree objects into one aggregated L-tree."""

    def __init__(self):
        self.root = {}
        self.stats = defaultdict(NodeStat)

    def _merge_dict(self, src, dst_path):
        for key, child in src.items():
            new_path = dst_path + (key,)
            dst_node = self._traverse_create(dst_path).setdefault(key, {})
            # Recurse first so we have dst_node for deeper children
            self._merge_dict(child, new_path)
        # nothing to return – tree built in place

    def _traverse_create(self, path):
        node = self.root
        for hop in path:
            node = node.setdefault(hop, {})
        return node

    def add_trace(self, trace_tree: TraceTree):
        # merge topology
        self._merge_dict(trace_tree.root, tuple())
        # merge latency stats
        for path, stat in trace_tree.stats.items():
            self.stats[path].latencies.extend(stat.latencies)

    ############### convenience output ###############
    def to_dot(self, outfile="ltree.dot"):
        """Dump L-tree in GraphViz DOT format."""
        def add_edges(node, path, out_lines):
            for child_key, child_node in node.items():
                child_path = path + (child_key,)
                edge_label = ""
                if self.stats[child_path].latencies:
                    edge_label = (
                        f" [label=\"p50={self.stats[child_path].p50:.1f}ms, "
                        f"n={len(self.stats[child_path].latencies)}\"]"
                    )
                out_lines.append(f"\"{'.'.join(map(str,path)) or 'root'}\" -> "
                                 f"\"{'.'.join(map(str,child_path))}\"{edge_label};")
                add_edges(child_node, child_path, out_lines)

        lines = ["digraph LTree {"]
        add_edges(self.root, tuple(), lines)
        lines.append("}")
        Path(outfile).write_text("\n".join(lines))
        print(f"[+] DOT file written to {outfile}")

###############################################################################
# 4. Driver code: load CASPER CSVs → build L-tree
###############################################################################
def build_ltree(casper_dir, limit=None):
    merger = LTreemerger()
    csv_files = sorted(Path(casper_dir).glob("*.csv"))
    for n, csv_path in enumerate(csv_files, 1):
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        for trace_id, df_trace in df.groupby("traceid"):
            ttree = TraceTree(df_trace)
            merger.add_trace(ttree)
        if limit and n >= limit:
            break
    return merger

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument("casper_dir", help="Folder with CASPER *.csv files")
    parser.add_argument("--out", default="ltree.dot",
                        help="Output DOT filename")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N csv files (for quick tests)")
    args = parser.parse_args()

    if not os.path.isdir(args.casper_dir):
        raise SystemExit("CASPER directory not found")

    ltree = build_ltree(args.casper_dir, limit=args.limit)
    ltree.to_dot(args.out)
    print("[✓] L-tree construction complete")