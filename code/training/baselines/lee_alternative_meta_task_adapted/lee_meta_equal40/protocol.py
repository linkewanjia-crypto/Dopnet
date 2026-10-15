"""Standard-library budget and episode contracts; no scientific execution."""
import math

EPOCHS = 40
FRESH_FRAMES = 12000
META_EPOCHS = 1
ONLINE_EPOCHS = 39
EPISODE_FRAMES = 32
SUPPORT_FRAMES = 16
QUERY_FRAMES = 16
UPDATES_PER_EPOCH = 375
DEADLINE_UTC = "2026-10-02T13:46:32.725968Z"


def phase_for_epoch(epoch):
    if type(epoch) is not int or not 1 <= epoch <= EPOCHS:
        raise ValueError("Global epoch must be a true integer in 1..40")
    return "add_meta" if epoch <= META_EPOCHS else "joint_online"


def episode_partition(doppler_hz):
    """Task adaptation: all 32 frame positions used once, without new samples.

    Training-only metadata sorts this existing generated batch. The lower
    Doppler half is support, the upper half query; ties use original position.
    Homogeneous batches have no Doppler shift, and must be reported as such.
    Neither metadata nor true labels enter the estimator's public forward.
    This is not the author's 1/2 kHz -> 1.5/2.5 kHz dataset.
    """
    if len(doppler_hz) != EPISODE_FRAMES:
        raise ValueError("A meta episode consumes exactly 32 fresh frames")
    values = [float(value) for value in doppler_hz]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Doppler labels must be finite and nonnegative")
    order = sorted(range(EPISODE_FRAMES), key=lambda i: (values[i], i))
    support, query = order[:SUPPORT_FRAMES], order[SUPPORT_FRAMES:]
    assert set(support).isdisjoint(query)
    assert sorted(support + query) == list(range(EPISODE_FRAMES))
    return support, query


def check_terminal_ledger(rows):
    if len(rows) != EPOCHS:
        raise ValueError("Training cannot qualify before 40 committed epochs")
    for epoch, row in enumerate(rows, 1):
        expected = {"global_epoch": epoch, "data_epoch": epoch,
                    "phase": phase_for_epoch(epoch),
                    "fresh_frames": FRESH_FRAMES,
                    "optimizer_updates": UPDATES_PER_EPOCH}
        for key, value in expected.items():
            if row.get(key) != value or (type(value) is int and type(row.get(key)) is not int):
                raise ValueError(f"Ledger epoch {epoch}: wrong {key}")
        if epoch == 1:
            if row.get("support_frames") != 6000 or row.get("query_frames") != 6000:
                raise ValueError("Meta support/query must partition the same 12000 frames")
        elif row.get("support_frames", 0) or row.get("query_frames", 0):
            raise ValueError("Online training must not add hidden support/query data")
    return {"committed_epochs": EPOCHS, "meta_epochs": META_EPOCHS,
            "online_epochs": ONLINE_EPOCHS, "fresh_frames": EPOCHS * FRESH_FRAMES,
            "ledger_only": True, "training_delivery_qualified": False}
