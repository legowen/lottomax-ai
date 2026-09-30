"""
Data ingestion: validate user-pasted official results and append them to
data/LOTTOMAX.csv safely (backup first, atomic replace).

No scraping — the user pastes official results.
"""
import csv
import os
import re
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from lotto_config import PICK, pool_for_date

CSV_NAME = "LOTTOMAX.csv"
HEADER = ('"PRODUCT","DRAW NUMBER","SEQUENCE NUMBER","DRAW DATE","NUMBER DRAWN 1",'
          '"NUMBER DRAWN 2","NUMBER DRAWN 3","NUMBER DRAWN 4","NUMBER DRAWN 5",'
          '"NUMBER DRAWN 6","NUMBER DRAWN 7","BONUS NUMBER"')
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DRAW_WEEKDAYS = (1, 4)  # Tuesday, Friday


def read_last_draw(data_dir: Path) -> dict:
    """Return {draw_number, date, all_numbers:set} from the main (SEQUENCE 0) rows."""
    path = Path(data_dir) / CSV_NAME
    last_no, last_date, seen = 0, "", set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["SEQUENCE NUMBER"].strip() != "0":
                continue
            no = int(row["DRAW NUMBER"])
            seen.add(no)
            if no > last_no:
                last_no, last_date = no, row["DRAW DATE"].strip()
    return {"draw_number": last_no, "date": last_date, "existing": seen}


def validate_new_draws(new: list, last_draw_number: int, last_date: str,
                       existing: Optional[set] = None) -> tuple:
    """
    Validate draws in order. Returns (valid_draws, errors); each error is
    {"row": 1-based index in `new`, "draw_number": ..., "error": str}.
    A rejected row does not advance the running "last" markers.
    """
    existing = existing or set()
    valid, errors = [], []
    cur_no, cur_date = last_draw_number, last_date

    def fail(i, d, msg):
        errors.append({"row": i, "draw_number": d.get("draw_number"), "error": msg})

    for i, d in enumerate(new, start=1):
        no, ds = d.get("draw_number"), d.get("date")
        nums, bonus = d.get("numbers"), d.get("bonus")

        if not isinstance(no, int) or isinstance(no, bool):
            fail(i, d, "회차 번호가 정수가 아닙니다"); continue
        if no in existing:
            fail(i, d, f"이미 존재하는 회차입니다 ({no})"); continue
        if no <= cur_no:
            fail(i, d, f"회차 번호는 마지막 회차({cur_no})보다 커야 합니다"); continue

        if not isinstance(ds, str) or not _DATE_RE.match(ds):
            fail(i, d, "날짜는 YYYY-MM-DD 형식이어야 합니다"); continue
        try:
            datetime.strptime(ds, "%Y-%m-%d")
        except ValueError:
            fail(i, d, f"존재하지 않는 날짜입니다 ({ds})"); continue
        if ds <= cur_date:
            fail(i, d, f"날짜는 마지막 회차 날짜({cur_date})보다 뒤여야 합니다"); continue

        pool = pool_for_date(ds)
        if (not isinstance(nums, list) or len(nums) != PICK
                or any(not isinstance(n, int) or isinstance(n, bool) for n in nums)):
            fail(i, d, f"번호는 정수 {PICK}개여야 합니다"); continue
        if len(set(nums)) != PICK:
            fail(i, d, "번호가 중복되었습니다"); continue
        if any(n < 1 or n > pool for n in nums):
            fail(i, d, f"번호는 1~{pool} 범위여야 합니다 ({ds} 기준)"); continue
        if not isinstance(bonus, int) or isinstance(bonus, bool) or not 1 <= bonus <= pool:
            fail(i, d, f"보너스는 1~{pool} 정수여야 합니다"); continue
        if bonus in nums:
            fail(i, d, "보너스가 본번호와 겹칩니다"); continue

        valid.append({"draw_number": no, "date": ds, "numbers": sorted(nums), "bonus": bonus})
        cur_no, cur_date = no, ds
    return valid, errors


def _format_row(d: dict) -> str:
    nums = ",".join(str(n) for n in sorted(d["numbers"]))
    return f'"LOTTO MAX",{d["draw_number"]},0,"{d["date"]}",{nums},{d["bonus"]}'


def append_draws_to_csv(valid_draws: list, data_dir: Path) -> Optional[Path]:
    """
    Back up the CSV to data/backup/LOTTOMAX_YYYYmmdd_HHMMSS.csv, write the new
    content to a temp file in the same directory, then os.replace() it in.
    Returns the backup path (None if there was nothing to add).
    """
    if not valid_draws:
        return None
    data_dir = Path(data_dir)
    src = data_dir / CSV_NAME
    original = src.read_text()

    backup_dir = data_dir / "backup"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = backup_dir / f"LOTTOMAX_{stamp}.csv"
    n = 1
    while backup.exists():  # two appends within the same second
        backup = backup_dir / f"LOTTOMAX_{stamp}_{n}.csv"
        n += 1
    backup.write_text(original)

    body = original if original.endswith("\n") else original + "\n"
    body += "\n".join(_format_row(d) for d in valid_draws) + "\n"

    fd, tmp = tempfile.mkstemp(dir=data_dir, prefix=".LOTTOMAX_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, src)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return backup


def estimate_missing_draws(last_date: str, today: Optional[date] = None) -> int:
    """Tue/Fri draws strictly between last_date and today (today's draw is not counted)."""
    today = today or date.today()
    d = date.fromisoformat(last_date) + timedelta(days=1)
    count = 0
    while d < today:
        if d.weekday() in DRAW_WEEKDAYS:
            count += 1
        d += timedelta(days=1)
    return count
