"""4-weekly counts of public GitHub commits with Claude co-author trailers, total then by model.

Usage: python runrate_nowcast/scripts/github_claude_commits.py runrate_nowcast/data/github_claude_commits.csv
Uses `gh api search/commits`. Unauthenticated search allows about one query a minute here;
with `gh auth login` it is roughly 30x faster. Counts carry incomplete_results flags from
GitHub (approximate when true). Writes the whole file again, most recent weeks first.
"""
import csv, json, subprocess, sys, time, datetime as dt
OUT = sys.argv[1]
REL = {"Opus 4": "2025-05-22", "Sonnet 4": "2025-05-22", "Opus 4.1": "2025-08-05", "Sonnet 4.5": "2025-09-29", "Haiku 4.5": "2025-10-15",
       "Opus 4.5": "2025-11-24", "Opus 4.6": "2026-02-04", "Sonnet 4.6": "2026-02-17", "Opus 4.7": "2026-04-16", "Opus 4.8": "2026-05-27",
       "Fable 5": "2026-06-09", "Sonnet 5": "2026-06-30", "Opus 5": "2026-07-24", "Fable 5.1": "2026-09-01", "Opus 5.5": "2026-09-22",
       "Sonnet 5.5": "2026-09-28", "Haiku 5.5": "2026-10-07", "Fable 5.2": "2026-11-09"}
def count(q):
    for i in range(8):
        r = subprocess.run(["gh", "api", "-X", "GET", "search/commits", "-f", f"q={q}", "-f", "per_page=1"], capture_output=True, text=True)
        try:
            d = json.loads(r.stdout)
            if "total_count" in d:
                return d["total_count"], d.get("incomplete_results")
        except Exception:
            pass
        time.sleep(60)
    return None, None
weeks = []
d = dt.date(2025, 2, 24)
while d + dt.timedelta(days=6) <= dt.date(2026, 11, 9):
    weeks.append(d); d += dt.timedelta(days=28)
weeks = weeks[::-1]                       # most recent first
jobs = [(w, "Claude") for w in weeks] + [(w, "Codex") for w in weeks[:6]]
jobs += [(w, f"Claude {m}") for w in weeks for m, r in REL.items()
         if dt.date.fromisoformat(r) <= w + dt.timedelta(days=6) and (w - dt.date.fromisoformat(r)).days < 300]
with open(OUT, "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["week_start", "query", "total_count", "incomplete"]); f.flush()
    for w, who in jobs:
        n, inc = count(f'"Co-Authored-By: {who}" committer-date:{w}..{w + dt.timedelta(days=6)}')
        wr.writerow([w, who, n, inc]); f.flush()
        time.sleep(10)
print("done", flush=True)
