# ltree_builder.py
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

###############################################################################
# 1. Utilities
###############################################################################
# NodeStat: stores request latencies and intervals
@dataclass
class NodeStat:
    latencies: list = field(default_factory=list)
    intervals: list = field(default_factory=list)   # (start, end) for child‑vs‑child overlap
    count: int = 0

    def update(self, rt, bt=None):
        self.count += 1
        if rt >= 0:
            self.latencies.append(rt)
            if bt is not None:
                self.intervals.append((bt, bt + rt))

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
# TraceTree: builds a tree for a single trace using rpcid hierarchy (0.1.1.2 …).
class TraceTree:
    """Builds a tree for a single trace using rpcid hierarchy (0.1.1.2 …)."""

    def __init__(self, df_trace):
        self.root = {}
        self.stats = defaultdict(NodeStat)
        self._build(df_trace)

    def _insert(self, path, rt, bt):
        node = self.root
        for hop in path:
            node = node.setdefault(hop, {})
        self.stats[tuple(path)].update(rt, bt)
        # record child intervals *only* when start time is available
        if bt is not None and len(path) > 1:
            parent = tuple(path[:-1])
            self._child_intervals[parent].append((bt, bt + rt))

    def _build(self, df_trace):
        self._child_intervals = defaultdict(list)
        # decide which column stores the RPC start time
        candidate_cols = ["bt", "timestamp", "cs", "startTime", "start", "ts"]
        start_col = next((c for c in candidate_cols if c in df_trace.columns), None)

        for _, row in df_trace.sort_values("rpcid").iterrows():
            raw_parts = str(row["rpcid"]).split(".")
            hops = []
            malformed = False
            for part in raw_parts:
                # handle weird tokens like '13‑84f9f…' → keep prefix before '-'
                num_token = part.split("-")[0]
                try:
                    hops.append(int(num_token))
                except ValueError:
                    malformed = True
                    break         # abandon this row; rpcid not integer‑hierarchy
            if malformed:
                continue           # skip bad rpcid rows silently
            bt_val = row.get(start_col) if start_col else None
            self._insert(hops, row["rt"], bt_val)

        # after we’ve seen all children we can decide parallel vs sequential
        self.parallel = {}
        for parent, ivals in self._child_intervals.items():
            if len(ivals) <= 1:
                self.parallel[parent] = False
                continue
            ivals.sort(key=lambda x: x[0])
            overlap = any(s < prev_end for (s, e), (_, prev_end)
                          in zip(ivals[1:], ivals))
            self.parallel[parent] = overlap

###############################################################################
# 3. L-tree merger
###############################################################################
# LTreemerger: merges many TraceTree objects into one aggregated L-tree.
class LTreemerger:
    """Merge many TraceTree objects into one aggregated L-tree."""

    def __init__(self):
        self.root = {}
        self.stats = defaultdict(NodeStat)
        # votes: how many traces said "children under P overlap?"
        self._par_votes: defaultdict[tuple, int] = defaultdict(int)
        self._par_total: defaultdict[tuple, int] = defaultdict(int)

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
        # 1 · merge topology
        self._merge_dict(trace_tree.root, tuple())

        # 2 · merge latency stats
        for path, stat in trace_tree.stats.items():
            self.stats[path].latencies.extend(stat.latencies)
            self.stats[path].count += stat.count

        # 3 · merge “is‑parallel?” votes
        for parent_path, is_par in trace_tree.parallel.items():
            self._par_votes[parent_path] += int(is_par)
            self._par_total[parent_path] += 1
    # ------------------------------------------------------------------
    def is_parallel(self, parent_path: tuple, threshold: float = 0.5) -> bool:
        """
        Return True if children under `parent_path` are considered parallel.

        We classify as parallel when ≥ `threshold` fraction of traces that
        contain `parent_path` had overlapping child intervals.
        """
        total = self._par_total.get(parent_path, 0)
        if total == 0:
            return False
        return self._par_votes[parent_path] / total >= threshold

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
# build_ltree: load CASPER CSVs → build L-tree
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