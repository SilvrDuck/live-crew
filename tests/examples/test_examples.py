"""Tests for examples in the examples/ folder.

This module automatically discovers and tests all examples by executing them
as separate Python processes and comparing their output against expected results.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


class TestExamplesExecution:
    """Automated test bed for all examples using expected output files."""

    @pytest.fixture
    def examples_dir(self):
        """Path to the examples directory."""
        return Path(__file__).parent.parent.parent / "examples"

    @pytest.fixture
    def test_examples_dir(self):
        """Path to the tests/examples directory."""
        return Path(__file__).parent

    @pytest.fixture
    def discovered_examples(self, examples_dir):
        """Discover all example directories."""
        return [
            d for d in examples_dir.iterdir() if d.is_dir() and (d / "main.py").exists()
        ]

    def test_examples_directory_exists(self, examples_dir):
        """Test that the examples directory exists."""
        assert examples_dir.exists(), "Examples directory should exist"
        assert examples_dir.is_dir(), "Examples should be a directory"

    def test_examples_are_discovered(self, discovered_examples):
        """Test that we can discover examples automatically."""
        assert len(discovered_examples) > 0, "Should discover at least one example"

        # Verify hello_world is discovered
        example_names = [ex.name for ex in discovered_examples]
        assert "hello_world" in example_names, "Should discover hello_world example"

    def test_all_examples_have_expected_files(
        self, discovered_examples, test_examples_dir
    ):
        """Test that every discovered example has a corresponding expected output file."""
        for example_dir in discovered_examples:
            expected_file = test_examples_dir / f"{example_dir.name}_expected.yaml"
            assert expected_file.exists(), (
                f"Example '{example_dir.name}' must have expected output file at {expected_file}"
            )

            # Validate the YAML file can be loaded
            try:
                with open(expected_file, "r") as f:
                    expected_data = yaml.safe_load(f)
                assert isinstance(expected_data, dict), (
                    f"Expected file for '{example_dir.name}' should contain a YAML dictionary"
                )
            except yaml.YAMLError as e:
                pytest.fail(
                    f"Expected file for '{example_dir.name}' contains invalid YAML: {e}"
                )

    def test_all_examples_have_required_files(self, discovered_examples):
        """Test that every discovered example has required files."""
        for example_dir in discovered_examples:
            # Check main.py exists (already checked in discovery)
            main_py = example_dir / "main.py"
            assert main_py.exists(), f"Example '{example_dir.name}' should have main.py"

            # Check README.md exists
            readme = example_dir / "README.md"
            assert readme.exists(), (
                f"Example '{example_dir.name}' should have README.md"
            )

    def test_example_execution_against_expected(
        self, discovered_examples, test_examples_dir
    ):
        """Test each example against its expected output file."""
        for example_dir in discovered_examples:
            self._test_single_example(example_dir, test_examples_dir)

    def _test_single_example(self, example_dir: Path, test_examples_dir: Path):
        """Test a single example against its expected output."""
        # Load expected output specification
        expected_file = test_examples_dir / f"{example_dir.name}_expected.yaml"
        with open(expected_file, "r") as f:
            expected = yaml.safe_load(f)

        main_py = example_dir / "main.py"
        timeout = expected.get("timeout", 30)
        expected_exit_code = expected.get("exit_code", 0)

        # Execute the example as a separate Python process
        try:
            result = subprocess.run(
                [sys.executable, str(main_py)],
                cwd=example_dir,  # Run from the example directory
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # Don't raise on non-zero exit
            )
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"Example '{example_dir.name}' timed out after {timeout} seconds"
            )

        # Check exit code
        assert result.returncode == expected_exit_code, (
            f"Example '{example_dir.name}' exited with code {result.returncode}, expected {expected_exit_code}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        output = result.stdout

        # Test 'contains' patterns
        if "contains" in expected:
            for pattern in expected["contains"]:
                assert pattern in output, (
                    f"Example '{example_dir.name}' output should contain '{pattern}'\n"
                    f"Actual output:\n{output}"
                )

        # Test count patterns
        if "counts" in expected:
            for pattern, expected_count in expected["counts"].items():
                actual_count = output.count(pattern)
                assert actual_count == expected_count, (
                    f"Example '{example_dir.name}' should contain '{pattern}' {expected_count} times, "
                    f"but found {actual_count} times\n"
                    f"Actual output:\n{output}"
                )

        # Test regex patterns
        if "regex_patterns" in expected:
            for regex_pattern in expected["regex_patterns"]:
                assert re.search(regex_pattern, output), (
                    f"Example '{example_dir.name}' output should match regex '{regex_pattern}'\n"
                    f"Actual output:\n{output}"
                )

    def test_all_examples_are_tested(self, discovered_examples, test_examples_dir):
        """Ensure that the test apparatus covers all discovered examples."""
        example_names = [ex.name for ex in discovered_examples]

        # Check that each example has an expected file
        for example_name in example_names:
            expected_file = test_examples_dir / f"{example_name}_expected.yaml"
            assert expected_file.exists(), (
                f"Example '{example_name}' needs an expected output file at {expected_file}"
            )

        # Check for orphaned expected files (expected files without corresponding examples)
        expected_files = list(test_examples_dir.glob("*_expected.yaml"))
        for expected_file in expected_files:
            example_name = expected_file.stem.replace("_expected", "")
            assert example_name in example_names, (
                f"Found orphaned expected file {expected_file} - no corresponding example '{example_name}'"
            )

        # Log discovered examples for visibility
        print(f"\nDiscovered examples: {example_names}")
        print(f"Expected files: {[f.name for f in expected_files]}")

        # This test passes if we discover examples and they all have expected files
        assert len(discovered_examples) > 0, "Should have examples to test"


class TestExampleInfrastructureValidation:
    """Validate the structure and consistency of examples."""

    @pytest.fixture
    def examples_dir(self):
        """Path to the examples directory."""
        return Path(__file__).parent.parent.parent / "examples"

    def test_example_directories_structure(self, examples_dir):
        """Test that example directories follow expected structure."""
        example_dirs = [d for d in examples_dir.iterdir() if d.is_dir()]

        for example_dir in example_dirs:
            # Each example should have a main.py
            main_py = example_dir / "main.py"
            if not main_py.exists():
                continue  # Skip directories without main.py (they're not examples)

            # Validate main.py structure
            main_content = main_py.read_text()
            assert 'if __name__ == "__main__":' in main_content, (
                f"Example '{example_dir.name}' main.py should have main guard"
            )
            assert "asyncio.run" in main_content or "async def" in main_content, (
                f"Example '{example_dir.name}' should use async patterns"
            )

            # Check for imports
            assert "live_crew" in main_content, (
                f"Example '{example_dir.name}' should import live_crew"
            )

    def test_expected_files_structure(self):
        """Test that expected files follow the correct YAML structure."""
        test_examples_dir = Path(__file__).parent
        expected_files = list(test_examples_dir.glob("*_expected.yaml"))

        assert len(expected_files) > 0, "Should have at least one expected file"

        for expected_file in expected_files:
            with open(expected_file, "r") as f:
                try:
                    expected_data = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(
                        f"Expected file {expected_file.name} contains invalid YAML: {e}"
                    )

            # Validate structure
            assert isinstance(expected_data, dict), (
                f"Expected file {expected_file.name} should contain a dictionary"
            )

            # Optional but recommended fields
            recommended_fields = ["contains", "exit_code", "timeout"]
            for field in recommended_fields:
                if field not in expected_data:
                    print(
                        f"Warning: {expected_file.name} missing recommended field '{field}'"
                    )

            # Validate field types if present
            if "contains" in expected_data:
                assert isinstance(expected_data["contains"], list), (
                    f"'contains' in {expected_file.name} should be a list"
                )

            if "counts" in expected_data:
                assert isinstance(expected_data["counts"], dict), (
                    f"'counts' in {expected_file.name} should be a dictionary"
                )

            if "regex_patterns" in expected_data:
                assert isinstance(expected_data["regex_patterns"], list), (
                    f"'regex_patterns' in {expected_file.name} should be a list"
                )

            if "exit_code" in expected_data:
                assert isinstance(expected_data["exit_code"], int), (
                    f"'exit_code' in {expected_file.name} should be an integer"
                )

            if "timeout" in expected_data:
                assert isinstance(expected_data["timeout"], (int, float)), (
                    f"'timeout' in {expected_file.name} should be a number"
                )
