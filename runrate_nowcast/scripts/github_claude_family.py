"""4-weekly public commit counts by Claude model family ('Co-Authored-By: Claude Opus' etc.)."""
import csv, json, subprocess, sys, time, datetime as dt
OUT = sys.argv[1]
def count(q):
    for i in range(8):
        r = subprocess.run(["gh", "api", "-X", "GET", "search/commits", "-f", f"q={q}", "-f", "per_page=1"], capture_output=True, text=True)
        try:
            d = json.loads(r.stdout)
            if "total_count" in d:
                return d["total_count"], d.get("incomplete_results")
        except Exception:
            pass
        time.sleep(180)
    return None, None
weeks = []
d = dt.date(2025, 2, 24)
while d + dt.timedelta(days=6) <= dt.date(2026, 11, 9):
    weeks.append(d); d += dt.timedelta(days=28)
weeks = weeks[::-1]
with open(OUT, "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["week_start", "query", "total_count", "incomplete"]); f.flush()
    for w in weeks:
        for fam in ("Opus", "Sonnet"):
            n, inc = count(f'"Co-Authored-By: Claude {fam}" committer-date:{w}..{w + dt.timedelta(days=6)}')
            wr.writerow([w, f"Claude {fam}", n, inc]); f.flush()
            time.sleep(90)
print("done", flush=True)
