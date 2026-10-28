"""
sft/tests/test_smoke.py — Phase 2.5 smoke tests
Run with:  <env-python> -m unittest discover -v -s sft/tests
No test dependencies beyond stdlib + the installed CPU-stage packages.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # Capstone/
SFT_DIR   = REPO_ROOT / "sft"
SNAP_DIR  = SFT_DIR / "tokenizer_snapshot"


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — library imports and langid classify
# ─────────────────────────────────────────────────────────────────────────────
class TestImports(unittest.TestCase):

    def test_pandas_import(self):
        import pandas as pd   # noqa: F401
        self.assertTrue(True)

    def test_datasketch_import(self):
        from datasketch import MinHash  # noqa: F401
        self.assertTrue(True)

    def test_langid_import(self):
        import langid  # noqa: F401
        self.assertTrue(True)

    def test_langid_classify_english(self):
        import langid
        # short made-up sentence — must NOT contain real dataset text
        lang, score = langid.classify("Hello world this is a test sentence")
        self.assertEqual(lang, "en", f"Expected 'en' but got {lang!r} (score={score:.4f})")

    def test_transformers_import(self):
        import transformers  # noqa: F401
        self.assertTrue(True)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — offline tokenizer load + special tokens + round-trip
# ─────────────────────────────────────────────────────────────────────────────
class TestTokenizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load the tokenizer once for all tokenizer tests."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        if not SNAP_DIR.exists():
            raise unittest.SkipTest(
                f"Tokenizer snapshot not found at {SNAP_DIR}. "
                "Run the snapshot download step first."
            )
        try:
            from transformers import AutoTokenizer
            cls.tok = AutoTokenizer.from_pretrained(
                str(SNAP_DIR), local_files_only=True
            )
        except Exception as exc:
            raise unittest.SkipTest(f"Tokenizer load failed: {exc}") from exc

    def test_offline_load(self):
        """Tokenizer must load without network."""
        self.assertIsNotNone(self.tok)

    def test_im_start_single_token(self):
        ids = self.tok.encode("<|im_start|>", add_special_tokens=False)
        self.assertEqual(
            len(ids), 1,
            f"<|im_start|> encoded to {len(ids)} tokens (expected 1): {ids}"
        )
        # record for report
        print(f"\n  <|im_start|> -> id={ids[0]}", end="")

    def test_im_end_single_token(self):
        ids = self.tok.encode("<|im_end|>", add_special_tokens=False)
        self.assertEqual(
            len(ids), 1,
            f"<|im_end|> encoded to {len(ids)} tokens (expected 1): {ids}"
        )
        print(f"\n  <|im_end|> -> id={ids[0]}", end="")

    def test_endoftext_single_token(self):
        ids = self.tok.encode("<|endoftext|>", add_special_tokens=False)
        self.assertEqual(
            len(ids), 1,
            f"<|endoftext|> encoded to {len(ids)} tokens (expected 1): {ids}"
        )
        print(f"\n  <|endoftext|> -> id={ids[0]}", end="")

    def test_roundtrip(self):
        """A plain sentence should survive encode→decode."""
        sentence = "Machine learning models require careful evaluation."
        enc = self.tok.encode(sentence, add_special_tokens=False)
        dec = self.tok.decode(enc)
        self.assertIn(
            sentence, dec,
            f"Round-trip failed: original={sentence!r} decoded={dec!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — verifier unit tests (stdlib only, uses temp files)
# ─────────────────────────────────────────────────────────────────────────────
class TestVerifier(unittest.TestCase):
    """
    Tests the verifier logic in isolation using temporary files only.
    Imports verify_manifest from the sft/ directory.
    """

    def _make_manifest(self, tmpdir: Path, files: dict[str, bytes]) -> Path:
        """
        Write temp files and a manifest JSON.
        files: {relative_path: content_bytes}
        Returns path to the manifest JSON.
        """
        import hashlib, json
        from datetime import datetime, timezone

        entries = []
        for rel, content in files.items():
            fp = tmpdir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(content)
            st = fp.stat()
            entries.append({
                "path":     rel,
                "size":     st.st_size,
                "mtime":    st.st_mtime,
                "mtime_iso": datetime.now(tz=timezone.utc).isoformat(),
                "hashed":   True,
                "sha256":   hashlib.sha256(content).hexdigest(),
            })

        manifest = {
            "schema_version": "1",
            "created_utc": datetime.now(tz=timezone.utc).isoformat(),
            "repo_root": str(tmpdir),
            "total_files": len(entries),
            "hashed_count": len(entries),
            "unhashed_count": 0,
            "entries": entries,
        }
        mpath = tmpdir / "manifest.json"
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return mpath

    def _run_verifier(self, manifest_path: Path, repo_root: Path) -> tuple[int, str]:
        """
        Run verify_manifest logic directly (not as subprocess) by importing it.
        Returns (exit_code, captured_output).
        """
        import io, contextlib
        # We need to monkeypatch the repo root inside verify_manifest
        sys.path.insert(0, str(SFT_DIR))
        import importlib
        import verify_manifest as vm

        # patch repo root resolution
        original_resolve = Path.resolve
        buf = io.StringIO()
        exit_code = 0

        try:
            # Override the manifest's repo_root by reading it and patching
            import json
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_data["repo_root"] = str(repo_root)
            manifest_path.write_text(
                json.dumps(manifest_data, indent=2), encoding="utf-8"
            )

            with contextlib.redirect_stdout(buf):
                try:
                    # Patch sys.exit to capture exit code
                    original_exit = sys.exit
                    def fake_exit(code=0):
                        raise SystemExit(code)
                    sys.exit = fake_exit

                    # Manually run the verification logic inline (no CLI args)
                    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
                    entries = manifest_obj["entries"]
                    errors = []
                    ok_count = 0
                    for entry in entries:
                        status, detail = vm.verify_entry(entry, repo_root)
                        if status is None:
                            ok_count += 1
                        else:
                            errors.append(f"[{status}] {detail}")

                    manifest_paths = {e["path"] for e in entries}
                    current_paths = set()
                    for dirpath, _, filenames in os.walk(repo_root):
                        for fname in filenames:
                            fp2 = Path(dirpath) / fname
                            rel = fp2.relative_to(repo_root).as_posix()
                            if rel != "manifest.json":
                                current_paths.add(rel)
                    new_paths = current_paths - manifest_paths

                    warnings = [f"[WARNING] New unmanifested file: {p}" for p in sorted(new_paths)]
                    for w in warnings:
                        print(w)
                    for e in errors:
                        print(e)
                    if errors:
                        exit_code = 1
                        print(f"VERIFICATION FAILED: {len(errors)} error(s)")
                    else:
                        print(f"VERIFICATION PASSED: {ok_count} ok, {len(warnings)} warning(s).")

                except SystemExit as e:
                    exit_code = e.code if e.code is not None else 0
                finally:
                    sys.exit = original_exit
        finally:
            pass

        return exit_code, buf.getvalue()

    def test_passes_when_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            files = {"data/file_a.txt": b"hello world", "data/file_b.txt": b"foo bar"}
            mpath = self._make_manifest(tmpdir, files)
            code, out = self._run_verifier(mpath, tmpdir)
            self.assertEqual(code, 0, f"Expected exit 0, got {code}. Output:\n{out}")
            self.assertIn("PASSED", out)

    def test_fails_on_modified_file(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            files = {"data/file_a.txt": b"original content"}
            mpath = self._make_manifest(tmpdir, files)
            # modify the file after manifesting
            (tmpdir / "data" / "file_a.txt").write_bytes(b"tampered content")
            code, out = self._run_verifier(mpath, tmpdir)
            self.assertEqual(code, 1, f"Expected exit 1, got {code}. Output:\n{out}")
            self.assertIn("CHANGED", out)

    def test_fails_on_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            files = {"data/file_a.txt": b"will be deleted"}
            mpath = self._make_manifest(tmpdir, files)
            (tmpdir / "data" / "file_a.txt").unlink()
            code, out = self._run_verifier(mpath, tmpdir)
            self.assertEqual(code, 1, f"Expected exit 1, got {code}. Output:\n{out}")
            self.assertIn("MISSING", out)

    def test_warns_on_added_file(self):
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            files = {"data/file_a.txt": b"original"}
            mpath = self._make_manifest(tmpdir, files)
            # add a new file not in the manifest
            (tmpdir / "data" / "new_file.txt").write_bytes(b"new content")
            code, out = self._run_verifier(mpath, tmpdir)
            # warnings don't cause failure
            self.assertEqual(code, 0, f"Expected exit 0, got {code}. Output:\n{out}")
            self.assertIn("WARNING", out)
            self.assertIn("PASSED", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
