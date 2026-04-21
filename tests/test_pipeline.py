"""Tests for the pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from epigraph.pipeline import PipelineState, main


class TestPipelineState:
    """Test pipeline state tracking."""

    def test_empty_state(self, tmp_path: Path) -> None:
        state = PipelineState("dev", state_path=tmp_path / "state.json")
        assert not state.is_completed("clinical")

    def test_record_and_check(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        state = PipelineState("dev", state_path=state_path)
        state.record_completion("clinical", 2.5)
        assert state.is_completed("clinical")
        assert not state.is_completed("annotations")
        # Verify persisted to disk
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert data["steps"]["clinical"]["status"] == "completed"
        assert data["steps"]["clinical"]["elapsed_s"] == 2.5

    def test_mode_mismatch_resets(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        # Write state for 'dev' mode
        state = PipelineState("dev", state_path=state_path)
        state.record_completion("clinical", 1.0)
        # Load with 'production' mode — should reset
        state2 = PipelineState("production", state_path=state_path)
        assert not state2.is_completed("clinical")

    def test_corrupted_file(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        state_path.write_text("not valid json{{{")
        state = PipelineState("dev", state_path=state_path)
        assert not state.is_completed("clinical")

    def test_print_status(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        state = PipelineState("dev", state_path=state_path)
        state.record_completion("clinical", 2.5)
        state.record_completion("annotations", 5.0)
        # Should not raise
        state.print_status()


class TestPipelineCLI:
    """Test the pipeline CLI."""

    def test_status_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--status"])
        assert result.exit_code == 0
        assert "Pipeline mode" in result.output

    def test_unknown_step(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--steps", "nonexistent"])
        # Should log error but not crash
        assert result.exit_code == 0

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "run-pipeline" in result.output or "pipeline" in result.output
