from lc0jax.training import downloads


def test_parse_training_index_and_latest():
    html = """
    <a href="training-run2--20250101-010101.tar">training-run2--20250101-010101.tar</a> 2025-01-01 01:01  2048
    <a href="training-run2--20250201-020202.tar">training-run2--20250201-020202.tar</a> 2025-02-01 02:02  4096
    <a href="training-run2--20250301-0317.tar">training-run2--20250301-0317.tar</a> 2025-03-01 03:17  1024
    """
    entries = downloads.parse_training_index(html)
    assert len(entries) == 3
    latest = downloads.latest_training_tar(entries)
    assert latest is not None
    assert latest.filename == "training-run2--20250301-0317.tar"
    assert latest.size == 1024


def test_pick_latest_training_tars_filters():
    html = """
    <a href="training-run1--20250101-010101.tar">training-run1--20250101-010101.tar</a> 2025-01-01 01:01  512
    <a href="training-run2--20250101-010101.tar">training-run2--20250101-010101.tar</a> 2025-01-01 01:01  2048
    <a href="training-run2--20250201-020202.tar">training-run2--20250201-020202.tar</a> 2025-02-01 02:02  4096
    """
    entries = downloads.parse_training_index(html)
    picks = downloads.pick_latest_training_tars(entries, count=1, min_size=1000, run=2)
    assert len(picks) == 1
    assert picks[0].filename == "training-run2--20250201-020202.tar"
