"""Tests for saved-output protection."""

import pytest

from src.utils.outputs import check_output_paths
def test_check_output_paths_allows_new_paths(tmp_path):
    # confirm unused output paths can be written normally
    check_output_paths([tmp_path / "new.csv"])

def test_check_output_paths_rejects_existing_paths(tmp_path):
    # ensure existing evidence is reported before it can be replaced
    first_path = tmp_path / "predictions.csv"
    second_path = tmp_path / "metrics_summary.csv"
    first_path.write_text("predictions", encoding="utf-8")
    second_path.write_text("metrics", encoding="utf-8")

    with pytest.raises(FileExistsError) as error:
        check_output_paths([first_path, second_path])

    assert str(first_path) in str(error.value)
    assert str(second_path) in str(error.value)
    assert "--overwrite" in str(error.value)

def test_check_output_paths_allows_explicit_overwrite(tmp_path):
    # make sure replacement is allowed only when it has been requested
    output_path = tmp_path / "existing.csv"
    output_path.write_text("old evidence", encoding="utf-8")

    check_output_paths([output_path], overwrite=True)