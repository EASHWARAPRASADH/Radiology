"""Training monitor: epoch and step parsing for classifier and detector logs."""

from __future__ import annotations

import yaml

from radio.monitor import parse_log, status


def test_classifier_log_gives_step_progress_and_eta(tmp_path):
    run = tmp_path / "mixed"
    run.mkdir()
    (run / "config.yaml").write_text(yaml.safe_dump({"optim": {"epochs": 4}}))
    log = tmp_path / "train.log"
    log.write_text("[data] train=10\n"
                   "  epoch 0 step 20/400 loss 0.3\n"
                   "[epoch 0] {'epoch': 0, 'train_loss': 0.3, 'val_mean_auroc': 0.7, 'seconds': 60.0}\n"
                   "  epoch 1 step 150/400 loss 0.2\n")
    s = status(run, log)
    assert s["total_epochs"] == 4 and len(s["epochs"]) == 1
    assert abs(s["current_epoch_fraction"] - 0.5) < 1e-9  # 150/400 of 4 epochs = 1.5 epochs, 1 finished
    assert s["eta_s"] is not None and s["process_alive"]  # log was just written
    assert s["done"] is None


def test_done_and_error_lines():
    epochs, done, error = parse_log("[epoch 0] {'epoch': 0, 'val_map25': 0.1, 'train_loss': 1.0, 'seconds': 5}\n"
                                    "[done] best val mAP@0.25=0.1\n")
    assert len(epochs) == 1 and done.startswith("best") and error is None
    assert parse_log("Traceback (most recent call last):\n")[2] is not None
