"""Failures that must not be reported as successful agent validation."""

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


docs = load_tool("check_docs")
runner = load_tool("run_tests")


class DocumentationValidationTests(unittest.TestCase):
    def test_missing_links_and_entry_points_fail_without_running_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "docs").mkdir()
            (root / "scripts").mkdir()
            (root / "scripts/exists.ps1").write_text("throw 'must not run'", encoding="utf-8")
            (root / "docs/target.md").write_text("# Target\n", encoding="utf-8")
            document = root / "docs/guide.md"
            document.write_text(
                "[valid](target.md#heading)\n[bad](missing.md)\n"
                "[remote](https://example.invalid/missing)\n"
                "[reference]: ../scripts/exists.ps1\n"
                "```powershell\n.\\scripts\\exists.ps1\n.\\scripts\\missing.ps1\n"
                "python -m geospatial_search.missing\n```\n"
                "`[literal](missing.md)`\n",
                encoding="utf-8",
            )
            errors = docs.check_document(document, root)
            self.assertEqual(len(errors), 3)
            self.assertTrue(any("missing link target missing.md" in error for error in errors))
            self.assertTrue(any("missing command" in error for error in errors))
            self.assertTrue(any("missing module" in error for error in errors))

    def test_untracked_notes_do_not_enter_default_doc_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "--quiet", str(root)], check=True)
            (root / "tracked.md").write_text("# Tracked\n", encoding="utf-8")
            (root / "untracked.md").write_text("[bad](missing.md)\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.md"], check=True)
            self.assertEqual(docs.tracked_markdown(root), [root / "tracked.md"])


class SyntheticValidationTests(unittest.TestCase):
    def test_skipped_or_empty_suite_is_not_a_pass(self):
        empty = unittest.TestResult()
        self.assertTrue(runner.skipped_are_errors(empty))
        skipped = unittest.TestResult()
        skipped.testsRun = 1
        skipped.skipped = [("FaissTest", "Optional search dependencies are required")]
        self.assertTrue(runner.skipped_are_errors(skipped))
        passed = unittest.TestResult()
        passed.testsRun = 1
        self.assertFalse(runner.skipped_are_errors(passed))


class PowerShellValidationTests(unittest.TestCase):
    def test_missing_native_tool_and_nonzero_exit_cannot_reuse_success_status(self):
        script = str(ROOT / "scripts/validate.ps1").replace("'", "''")
        harness = (
            "$tokens=$null; $errors=$null; "
            f"$ast=[System.Management.Automation.Language.Parser]::ParseFile('{script}',[ref]$tokens,[ref]$errors); "
            "$node=$ast.Find({param($item) $item -is [System.Management.Automation.Language.FunctionDefinitionAst] "
            "-and $item.Name -eq 'Invoke-CheckedNative'},$true); "
            "${function:Invoke-TestNative}=$node.Body.GetScriptBlock(); $global:LASTEXITCODE=0; "
        )
        for invocation in (
            "Invoke-TestNative -Executable 'missing-tool-for-validation-regression' -Arguments @()",
            "Invoke-TestNative -Executable 'powershell.exe' -Arguments @('-NoProfile','-Command','exit 7')",
        ):
            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    harness + f"try {{ {invocation}; exit 99 }} catch {{ exit 0 }}",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_setup_refuses_redirected_environment_before_dependency_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            script = root / "scripts/setup-dev.ps1"
            script.write_bytes((ROOT / "scripts/setup-dev.ps1").read_bytes())
            target = root / "live-environment"
            target.mkdir()
            marker = target / "preserve.txt"
            marker.write_text("live environment must not change", encoding="utf-8")
            redirect = root / ".venv-dev"
            # Junctions need no developer-mode privilege on Windows CI.
            result = subprocess.run(
                ["cmd.exe", "/c", "mklink", "/J", str(redirect), str(target)],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertTrue(os.path.isdir(redirect), result.stdout + result.stderr)
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(script)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("redirected environment", result.stdout + result.stderr)
            self.assertEqual(list(target.iterdir()), [marker])
            self.assertEqual(marker.read_text(encoding="utf-8"), "live environment must not change")


if __name__ == "__main__":
    unittest.main()
