import pandas as pd

from powerfc.cli import main


def test_cli_ingest_and_backtest(tmp_path):
    csv = tmp_path / "data.csv"
    assert main(["ingest", "--source", "synthetic", "--days", "120", "--out", str(csv)]) == 0
    assert csv.exists()

    out = tmp_path / "reports"
    rc = main(
        [
            "backtest",
            "--source",
            "csv",
            "--path",
            str(csv),
            "--models",
            "naive",
            "ridge",
            "--test-days",
            "7",
            "--train-window",
            "60",
            "--out-dir",
            str(out),
        ]
    )
    assert rc == 0
    metrics = pd.read_csv(out / "metrics.csv", index_col="model")
    assert list(metrics.index) == ["naive", "ridge"]
    assert (out / "predictions.csv").exists()
