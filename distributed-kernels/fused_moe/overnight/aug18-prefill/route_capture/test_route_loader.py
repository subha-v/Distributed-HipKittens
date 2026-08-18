#!/usr/bin/env python3
"""CPU-only unit test for the captured-route replay loader.

Builds fixtures through the SAME writer the serving hook uses
(route_replay.write_capture_npz -- the hook holds a byte-identical copy of
those keys, and test_writer_key_set_matches_hook pins them against the hook
source), loads them back through the harness loader, and asserts the four
properties the decisive experiment rests on:

  1. shapes/dtypes the harness can copy straight into its symmetric route
     buffers ([N,tokens,topk] int32 + float32, ids in range);
  2. determinism -- the same file and args give byte-identical stacks, so a
     captured arm re-run days later replays the same chunks;
  3. shuffle invariance -- "shuffled" preserves the routed-row MULTISET
     exactly (every expert count, every destination total) while changing the
     token order, which is what makes captured-vs-shuffled a clean isolation
     of run correlation;
  4. the guards -- wrong shape, out-of-range ids, wrong format version, bad
     order, and empty layer filters fail loudly instead of silently
     measuring the wrong thing.

Run:  python3 test_route_loader.py      (or python3 -m unittest test_route_loader)
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mok_patch", "mok_synthetic_prefill"
    ),
)

import route_replay  # noqa: E402

TOKENS = 128
TOPK = 8
EXPERTS = 256
WORLD = 8


def _correlated_ids(n_calls, tokens=TOKENS, topk=TOPK, experts=EXPERTS, seed=11):
    """Run-correlated routes: contiguous spans of rows share a hot expert set.

    This is the property the shuffled arm destroys, so the fixture must have
    it or the shuffle test proves nothing.
    """

    rng = np.random.default_rng(seed)
    out = np.zeros((n_calls, tokens, topk), dtype=np.int32)
    for c in range(n_calls):
        row = 0
        while row < tokens:
            span = int(min(rng.integers(8, 24), tokens - row))
            hot = rng.choice(experts, size=topk * 2, replace=False)
            for r in range(row, row + span):
                out[c, r] = rng.permutation(hot)[:topk]
            row += span
    return out


def _write_fixture(path, n_calls=6, **kwargs):
    ids = _correlated_ids(n_calls, **kwargs)
    rng = np.random.default_rng(3)
    wgt = rng.random(ids.shape, dtype=np.float32)
    wgt /= wgt.sum(axis=2, keepdims=True)
    route_replay.write_capture_npz(
        path,
        ids,
        wgt,
        [f"router{i:03d}" for i in range(n_calls)],
        call_index=np.zeros(n_calls, dtype=np.int32),
        num_experts=EXPERTS,
        rank=0,
        pid=4242,
    )
    return ids, wgt


class RouteLoaderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.path = os.path.join(self.dir, "routes_rank0_pid4242.npz")
        self.ids, self.wgt = _write_fixture(self.path)
        route_replay.load_capture.cache_clear()

    def tearDown(self):
        route_replay.load_capture.cache_clear()
        self._tmp.cleanup()

    def _load(self, order="captured", **kw):
        args = dict(
            path=self.path, rank=0, tokens=TOKENS, topk=TOPK,
            num_experts=EXPERTS, world_size=WORLD, order=order, seed=1234,
        )
        args.update(kw)
        return route_replay.load_capture(**args)

    # ---- 1. shapes / dtypes / values ----
    def test_shapes_and_dtypes(self):
        cap = self._load()
        self.assertEqual(cap.topk_ids.shape, (6, TOKENS, TOPK))
        self.assertEqual(cap.topk_weights.shape, (6, TOKENS, TOPK))
        self.assertEqual(cap.topk_ids.dtype, np.int32)
        self.assertEqual(cap.topk_weights.dtype, np.float32)
        self.assertEqual(cap.calls, 6)
        self.assertEqual(len(cap.layers), 6)
        self.assertEqual(len(cap.call_index), 6)
        self.assertGreaterEqual(int(cap.topk_ids.min()), 0)
        self.assertLess(int(cap.topk_ids.max()), EXPERTS)

    def test_captured_order_is_verbatim(self):
        cap = self._load()
        np.testing.assert_array_equal(cap.topk_ids, self.ids.astype(np.int32))
        np.testing.assert_allclose(
            cap.topk_weights, self.wgt.astype(np.float16).astype(np.float32)
        )

    def test_arrays_are_read_only(self):
        cap = self._load()
        with self.assertRaises(ValueError):
            cap.topk_ids[0, 0, 0] = 1

    # ---- 2. determinism ----
    def test_determinism_across_loads(self):
        a = self._load("shuffled")
        route_replay.load_capture.cache_clear()
        b = self._load("shuffled")
        self.assertIsNot(a, b)
        np.testing.assert_array_equal(a.topk_ids, b.topk_ids)
        np.testing.assert_array_equal(a.topk_weights, b.topk_weights)
        self.assertEqual(a.metadata["route_ids_sha256"], b.metadata["route_ids_sha256"])

    def test_cache_returns_the_same_object(self):
        # synthetic_inputs (setup route) and the harness (replay stack) must
        # not be able to disagree about what "call 0" is.
        self.assertIs(self._load(), self._load())

    def test_seed_changes_the_permutation(self):
        a = self._load("shuffled", seed=1234)
        route_replay.load_capture.cache_clear()
        b = self._load("shuffled", seed=99)
        self.assertFalse(np.array_equal(a.topk_ids, b.topk_ids))

    # ---- 3. shuffle invariance: same multiset, different order ----
    def test_shuffle_preserves_route_multiset(self):
        cap = self._load()
        shf = self._load("shuffled")
        self.assertEqual(
            cap.metadata["route_multiset_sha256"],
            shf.metadata["route_multiset_sha256"],
        )
        for c in range(cap.calls):
            np.testing.assert_array_equal(
                np.bincount(cap.topk_ids[c].reshape(-1), minlength=EXPERTS),
                np.bincount(shf.topk_ids[c].reshape(-1), minlength=EXPERTS),
            )
        self.assertFalse(np.array_equal(cap.topk_ids, shf.topk_ids))

    def test_shuffle_is_a_row_permutation(self):
        cap = self._load()
        shf = self._load("shuffled")
        for c in range(cap.calls):
            src = sorted(map(tuple, cap.topk_ids[c].tolist()))
            dst = sorted(map(tuple, shf.topk_ids[c].tolist()))
            self.assertEqual(src, dst)          # whole rows travel intact
            perm = route_replay._permutation(1234, c, TOKENS)
            np.testing.assert_array_equal(shf.topk_ids[c], cap.topk_ids[c][perm])
            np.testing.assert_array_equal(shf.topk_weights[c], cap.topk_weights[c][perm])

    def test_shuffle_lowers_within_chunk_concentration(self):
        # The fixture is run-correlated; a permutation must reduce the spread
        # of destination load across row BLOCKS while leaving the call total
        # untouched. If this ever fails, the two arms are not separating what
        # the experiment claims they separate.
        def block_spread(ids):
            dest = ids.reshape(-1, TOPK) // (EXPERTS // WORLD)
            blocks = dest.reshape(8, -1)
            per = np.stack(
                [np.bincount(b, minlength=WORLD) for b in blocks]
            ).astype(np.float64)
            return float((per.max(axis=1) / per.sum(axis=1)).mean())

        cap = self._load()
        shf = self._load("shuffled")
        self.assertGreater(
            np.mean([block_spread(cap.topk_ids[c]) for c in range(cap.calls)]),
            np.mean([block_spread(shf.topk_ids[c]) for c in range(shf.calls)]),
        )

    # ---- metadata / stats ----
    def test_metadata_records_provenance_and_skew(self):
        cap = self._load()
        meta = cap.metadata
        self.assertEqual(meta["route_order"], "captured")
        self.assertEqual(meta["route_calls"], 6)
        self.assertTrue(meta["route_file_is_own_rank"])
        self.assertEqual(len(meta["route_file_sha256"]), 64)
        skew = meta["route_skew"]
        for key in (
            "max_rank_x_uniform_p50", "max_rank_x_uniform_p95",
            "max_rank_x_uniform_max", "max_rank_x_uniform_aggregate",
            "expert_gini_aggregate", "top1_expert_x_uniform", "experts_used",
        ):
            self.assertIn(key, skew)
        self.assertGreaterEqual(skew["max_rank_x_uniform_max"], skew["max_rank_x_uniform_p50"])
        self.assertGreaterEqual(skew["max_rank_x_uniform_p50"], 1.0)

    # ---- directory resolution ----
    def test_directory_picks_this_ranks_file(self):
        other = os.path.join(self.dir, "routes_rank3_pid7.npz")
        _write_fixture(other, n_calls=2)
        route_replay.load_capture.cache_clear()
        cap = route_replay.load_capture(
            self.dir, 3, TOKENS, TOPK, EXPERTS, WORLD, "captured", 1234
        )
        self.assertEqual(cap.calls, 2)
        self.assertEqual(os.path.basename(cap.metadata["route_file"]), "routes_rank3_pid7.npz")

    def test_directory_falls_back_when_rank_is_missing(self):
        cap = route_replay.load_capture(
            self.dir, 5, TOKENS, TOPK, EXPERTS, WORLD, "captured", 1234
        )
        self.assertFalse(cap.metadata["route_file_is_own_rank"])

    def test_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            route_replay.load_capture(
                os.path.join(self.dir, "nope"), 0, TOKENS, TOPK, EXPERTS, WORLD
            )

    # ---- filters ----
    def test_layer_filter_and_max_calls(self):
        cap = self._load(layers=("router001", "router004"))
        self.assertEqual(cap.calls, 2)
        self.assertEqual(cap.layers, ("router001", "router004"))
        route_replay.load_capture.cache_clear()
        cap2 = self._load(max_calls=3)
        self.assertEqual(cap2.calls, 3)

    def test_unknown_layer_filter_raises(self):
        with self.assertRaises(ValueError):
            self._load(layers=("router999",))

    # ---- guards ----
    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            self._load(tokens=TOKENS + 1)

    def test_bad_order_raises(self):
        with self.assertRaises(ValueError):
            self._load(order="reversed")

    def test_out_of_range_ids_raise(self):
        bad = os.path.join(self.dir, "routes_rank9_pid1.npz")
        ids = _correlated_ids(2)
        ids[0, 0, 0] = 255
        wgt = np.ones(ids.shape, dtype=np.float32) / TOPK
        # Declare 128 experts in the file so the shape/scalar checks all agree
        # and the RANGE guard is the one under test.
        route_replay.write_capture_npz(
            bad, ids % 128, wgt, ["a", "b"], num_experts=128, rank=9
        )
        with np.load(bad, allow_pickle=False) as blob:
            keep = {k: blob[k] for k in blob.files}
        keep["topk_ids"] = keep["topk_ids"].copy()
        keep["topk_ids"][0, 0, 0] = 200          # >= the declared 128
        np.savez(bad, **keep)
        with self.assertRaises(ValueError) as ctx:
            route_replay.load_capture(bad, 9, TOKENS, TOPK, 128, WORLD)
        self.assertIn("out of range", str(ctx.exception))

    def test_missing_key_raises(self):
        broken = os.path.join(self.dir, "routes_rank8_pid1.npz")
        with np.load(self.path, allow_pickle=False) as blob:
            keep = {k: blob[k] for k in blob.files if k != "layers"}
        np.savez(broken, **keep)
        with self.assertRaises(ValueError) as ctx:
            route_replay.load_capture(broken, 8, TOKENS, TOPK, EXPERTS, WORLD)
        self.assertIn("layers", str(ctx.exception))

    def test_format_version_mismatch_raises(self):
        future = os.path.join(self.dir, "routes_rank7_pid1.npz")
        with np.load(self.path, allow_pickle=False) as blob:
            keep = {k: blob[k] for k in blob.files}
        keep["format_version"] = np.int32(route_replay.CAPTURE_FORMAT_VERSION + 1)
        np.savez(future, **keep)
        with self.assertRaises(ValueError):
            route_replay.load_capture(future, 7, TOKENS, TOPK, EXPERTS, WORLD)

    # ---- writer/hook agreement ----
    def test_writer_key_set_matches_hook(self):
        """The hook writes its own savez; drift there is silent data loss."""

        hook = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "skewhook_v2", "sitecustomize.py"
        )
        with open(hook, "r", encoding="utf-8") as handle:
            src = handle.read()
        body = src[src.index("payload = dict(", src.index("def dump_routes")):]
        body = body[: body.index("with open(tmp")]
        keys = set(re.findall(r"^\s{16}(\w+)=", body, flags=re.M))
        self.assertEqual(keys, set(route_replay.CAPTURE_KEYS))
        with np.load(self.path, allow_pickle=False) as blob:
            self.assertEqual(set(blob.files), set(route_replay.CAPTURE_KEYS))

    def test_hook_capture_bound_is_the_documented_one(self):
        hook = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "skewhook_v2", "sitecustomize.py"
        )
        with open(hook, "r", encoding="utf-8") as handle:
            src = handle.read()
        self.assertIn('os.environ.get("M15_ROUTE_CAPTURE_MAX", "64")', src)
        self.assertIn('os.environ.get("M15_ROUTE_CAPTURE_TOKENS", "4096")', src)
        self.assertIn('os.environ.get("M15_ROUTE_CAPTURE_FLUSH", "16")', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
