"""Data ingest tests. Everything runs on temporary copies — the real data/ is never written."""
import hashlib
import os
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import app as lotto
import data_ingest as di

REAL_CSV = Path(__file__).parent.parent.parent / "data" / "LOTTOMAX.csv"
HEADER = di.HEADER
BASE_ROWS = [
    '"LOTTO MAX",1,0,"2026-09-15",1,2,3,4,5,6,7,8',
    '"LOTTO MAX",1,1,"2026-09-15",9,10,11,12,13,14,15,0',   # Maxmillions line: not a main draw
    '"LOTTO MAX",2,0,"2026-09-18",5,10,15,20,25,30,35,8',
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def mini(tmp_path):
    (tmp_path / "LOTTOMAX.csv").write_text(HEADER + "\n" + "\n".join(BASE_ROWS) + "\n")
    return tmp_path


def draw(no, ds, nums=(3, 9, 14, 22, 31, 40, 47), bonus=12):
    return {"draw_number": no, "date": ds, "numbers": list(nums), "bonus": bonus}


# ---------------- validation ----------------

def test_valid_draw_accepted_and_sorted():
    valid, errors = di.validate_new_draws([draw(3, "2026-09-22", (47, 3, 40, 9, 14, 31, 22))], 2, "2026-09-18")
    assert errors == []
    assert valid[0]["numbers"] == [3, 9, 14, 22, 31, 40, 47]


def test_duplicate_draw_number_rejected():
    valid, errors = di.validate_new_draws([draw(2, "2026-09-22")], 2, "2026-09-18", existing={1, 2})
    assert valid == [] and "이미 존재" in errors[0]["error"] and errors[0]["row"] == 1


def test_duplicate_within_batch_rejected():
    valid, errors = di.validate_new_draws(
        [draw(3, "2026-09-22"), draw(3, "2026-09-25")], 2, "2026-09-18")
    assert len(valid) == 1 and errors[0]["row"] == 2


@pytest.mark.parametrize("nums,bonus,frag", [
    ((3, 3, 14, 22, 31, 40, 47), 12, "중복"),
    ((3, 9, 14, 22, 31, 40), 12, "7개"),
    ((0, 9, 14, 22, 31, 40, 47), 12, "범위"),
    ((3, 9, 14, 22, 31, 40, 53), 12, "범위"),
    ((3, 9, 14, 22, 31, 40, 47), 0, "보너스"),
    ((3, 9, 14, 22, 31, 40, 47), 47, "겹칩니다"),
])
def test_bad_numbers_rejected(nums, bonus, frag):
    valid, errors = di.validate_new_draws([draw(3, "2026-09-22", nums, bonus)], 2, "2026-09-18")
    assert valid == [] and frag in errors[0]["error"]


def test_pool_depends_on_date():
    # 52 is only legal from the 7/52 era (2026-04-14)
    assert di.validate_new_draws([draw(3, "2026-04-10", (3, 9, 14, 22, 31, 40, 52))], 2, "2026-04-03")[0] == []
    assert di.validate_new_draws([draw(3, "2026-04-14", (3, 9, 14, 22, 31, 40, 52))], 2, "2026-04-03")[0] != []


@pytest.mark.parametrize("ds", ["2026-09-18", "2026-09-10", "26-09-22", "2026/09/22", "2026-02-30"])
def test_bad_or_backwards_date_rejected(ds):
    valid, errors = di.validate_new_draws([draw(3, ds)], 2, "2026-09-18")
    assert valid == [] and errors


def test_draw_number_must_increase():
    valid, errors = di.validate_new_draws([draw(1, "2026-09-22")], 2, "2026-09-18")
    assert valid == [] and "커야" in errors[0]["error"]


def test_rejected_row_does_not_advance_markers():
    valid, errors = di.validate_new_draws(
        [draw(3, "2026-09-22", (1, 1, 2, 3, 4, 5, 6)), draw(3, "2026-09-22")], 2, "2026-09-18")
    assert [v["draw_number"] for v in valid] == [3] and errors[0]["row"] == 1


# ---------------- writing ----------------

def test_append_backs_up_and_appends(mini):
    before = (mini / "LOTTOMAX.csv").read_text()
    valid, _ = di.validate_new_draws(
        [draw(3, "2026-09-22", (47, 3, 40, 9, 14, 31, 22))], 2, "2026-09-18")
    backup = di.append_draws_to_csv(valid, mini)
    assert backup.parent == mini / "backup" and backup.name.startswith("LOTTOMAX_")
    assert backup.read_text() == before                      # backup is the untouched original
    after = (mini / "LOTTOMAX.csv").read_text()
    assert after.startswith(before)
    assert after.splitlines()[-1] == '"LOTTO MAX",3,0,"2026-09-22",3,9,14,22,31,40,47,12'
    assert not [p for p in mini.iterdir() if p.suffix == ".tmp"]
    last = di.read_last_draw(mini)
    assert last["draw_number"] == 3 and last["date"] == "2026-09-22"


def test_two_appends_same_second_keep_both_backups(mini):
    a, _ = di.validate_new_draws([draw(3, "2026-09-22")], 2, "2026-09-18")
    b, _ = di.validate_new_draws([draw(4, "2026-09-25")], 3, "2026-09-22")
    b1 = di.append_draws_to_csv(a, mini)
    b2 = di.append_draws_to_csv(b, mini)
    assert b1 != b2 and b1.exists() and b2.exists()


def test_write_is_atomic_on_failure(mini, monkeypatch):
    orig = sha(mini / "LOTTOMAX.csv")
    valid, _ = di.validate_new_draws([draw(3, "2026-09-22")], 2, "2026-09-18")

    def boom(*a, **k):
        raise OSError("disk exploded")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        di.append_draws_to_csv(valid, mini)
    assert sha(mini / "LOTTOMAX.csv") == orig                 # original never half-written
    assert not [p for p in mini.iterdir() if p.suffix == ".tmp"]


def test_append_nothing_is_noop(mini):
    assert di.append_draws_to_csv([], mini) is None
    assert not (mini / "backup").exists()


def test_estimate_missing_draws():
    # last draw Fri 2026-09-25: Tue 09-29 has happened by Wed 09-30
    assert di.estimate_missing_draws("2026-09-25", date(2026, 9, 30)) == 1
    # Fri 10-02 is "today": not counted yet
    assert di.estimate_missing_draws("2026-09-25", date(2026, 10, 2)) == 1
    assert di.estimate_missing_draws("2026-09-25", date(2026, 10, 3)) == 2
    assert di.estimate_missing_draws("2026-09-25", date(2026, 9, 26)) == 0
    assert di.estimate_missing_draws("2026-02-20", date(2026, 9, 29)) == 62


# ---------------- API on a temporary data dir ----------------

@pytest.fixture
def temp_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    (tmp_path / "LOTTOMAX.csv").write_text(REAL_CSV.read_text())
    monkeypatch.setattr(lotto, "DATA_DIR", tmp_path)
    real_before = sha(REAL_CSV)
    with TestClient(lotto.app) as c:
        yield c, tmp_path
    monkeypatch.undo()
    lotto.load_csv_data()                                       # restore real state for other tests
    assert sha(REAL_CSV) == real_before                        # real data untouched


def test_api_data_status(temp_api):
    c, _ = temp_api
    r = c.get("/data/status").json()
    assert r["last_draw_number"] == 1273 and r["last_draw_date"] == "2026-09-25"
    assert r["total_draws"] == 1273 and r["era2_draws"] == 770
    assert r["days_since_last"] >= 0 and r["estimated_missing_draws"] >= 0


def test_api_append_updates_stats_and_invalidates(temp_api):
    c, d = temp_api
    lotto.state["backtest"] = {"stale": True}
    lotto.state["signal_lab"] = {"stale": True}
    body = {"draws": [{"draw_number": 1274, "date": "2026-09-29",
                       "numbers": [52, 3, 9, 14, 22, 31, 40], "bonus": 12}]}
    r = c.post("/data/append", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["added"] == 1 and out["total_draws"] == 1274 and "LSTM 재학습 권장" in out["recommendation"]
    assert lotto.state["backtest"] is None and lotto.state["signal_lab"] is None
    assert len(lotto.state["main_draws"]) == 771
    assert lotto.state["main_draws"][-1] == [3, 9, 14, 22, 31, 40, 52]
    assert c.get("/").json()["main_draws"] == 771
    assert (d / "backup").exists() and len(list((d / "backup").iterdir())) == 1
    assert c.get("/data/status").json()["last_draw_number"] == 1274


def test_api_append_rejects_duplicates_and_bad_rows(temp_api):
    c, d = temp_api
    before = sha(d / "LOTTOMAX.csv")
    dup = {"draws": [{"draw_number": 1273, "date": "2026-09-29", "numbers": [3, 9, 14, 22, 31, 40, 47], "bonus": 12}]}
    r = c.post("/data/append", json=dup)
    assert r.status_code == 400 and "이미 존재" in r.json()["detail"]
    assert sha(d / "LOTTOMAX.csv") == before and not (d / "backup").exists()


def test_api_append_partial(temp_api):
    c, _ = temp_api
    body = {"draws": [
        {"draw_number": 1274, "date": "2026-09-29", "numbers": [3, 9, 14, 22, 31, 40, 47], "bonus": 12},
        {"draw_number": 1275, "date": "2026-10-02", "numbers": [3, 3, 14, 22, 31, 40, 47], "bonus": 12},
    ]}
    out = c.post("/data/append", json=body).json()
    assert out["added"] == 1 and out["rejected"][0]["row"] == 2


def test_api_append_malformed_is_422(temp_api):
    c, _ = temp_api
    assert c.post("/data/append", json={"draws": []}).status_code == 422
    assert c.post("/data/append", json={"draws": [{"draw_number": "x"}]}).status_code == 422
