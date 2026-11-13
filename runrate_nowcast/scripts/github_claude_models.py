"""Weekly public GitHub commit counts by Claude model, from the Claude Code co-author trailer.

Usage (from the repo root, with `gh auth login` done):
    python runrate_nowcast/scripts/github_claude_models.py runrate_nowcast/data/github_claude_models.csv [plan]

Windows: the 4-weekly 7-day windows of github_claude_commits.csv (Mondays 2025-02-24 + 28k).
plan = "price" (default): what the price index needs, from 2025-11-03 on: the Haiku and Fable family
                counts, and in each family the two newest models when their list or cache-read
                prices differ (e.g. Opus 4.5 vs 4.1, Sonnet 5 vs 4.6, Opus 5.5 vs 5, Haiku 5.5 vs 4.5).
       "first": weekly totals / Opus / Sonnet 2025-10-27..2025-12-29 (off the 4-weekly grid) and
                4-weekly totals / Opus / Sonnet 2025-04..2025-10: when model names entered the trailer.
       "models": the two newest models of every family in every window (per-model detail).
       "2w":    family counts and the two newest models per family on the windows half-way between.

Query: '"Co-Authored-By: Claude <Model>" committer-date:<Mon>..<Sun>'. GitHub's phrase search is a
token prefix match: "Claude Opus 5" also matches "Claude Opus 5.5 ...". Counts are approximate
(incomplete_results), and two runs of the same query differ by ~10%, so subtracting a child's count
from a parent's is useless (2026-11-02: "Opus 5" minus "Opus 5.5" gave 2.8M; the exact count is
0.15M). A model whose name is a prefix of a released model is therefore queried with the next
token of the trailer, the e-mail: '"Co-Authored-By: Claude Opus 5 noreply"' matches
"Claude Opus 5 <noreply@anthropic.com>" and not "Claude Opus 5.5 <...>". Such rows are stored with
query "Claude <Model> noreply". pricing.model_counts uses these exact counts.

Commit search is expensive for GitHub, so its secondary rate limit trips after a few quick
queries even when authenticated (about one query a minute is sustainable). This script waits PACE
seconds between queries, backs off on a 403, and appends each result as it arrives, so it can be
stopped and resumed (queries already in this file or in the family / totals files are skipped).
"""
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time

OUT = sys.argv[1]
PLAN = sys.argv[2] if len(sys.argv) > 2 else "price"
PACE = float(os.environ.get("PACE", "20"))
FIRST = dt.date(2025, 2, 24)
LAST_END = dt.date(2026, 11, 8)
NAMES_FROM = dt.date(2025, 11, 3)          # first window with model names in the trailer (see plan "first")
FAMILIES = ("Opus", "Sonnet", "Haiku", "Fable")
MAX_AGE = 400


def _prices():
    path = os.path.join(os.path.dirname(os.path.abspath(OUT)), "model_prices.csv")
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            fam, ver = r["family"].capitalize(), r["model"].split("-")[-1]
            out[f"{fam} {ver}"] = (dt.date.fromisoformat(r["release_date"]), float(r["output_usd_mtok"]),
                                   float(r["cache_read_usd_mtok"]) / float(r["input_usd_mtok"]))
    return out


PRICES = _prices()


def windows(plan):
    w4 = []
    d = FIRST
    while d + dt.timedelta(days=6) <= LAST_END:
        w4.append(d)
        d += dt.timedelta(days=28)
    if plan == "2w":
        return [w + dt.timedelta(days=14) for w in w4 if w + dt.timedelta(days=20) <= LAST_END]
    return w4


def released(w):
    end = w + dt.timedelta(days=6)
    return [m for m, (r, _, _) in PRICES.items() if r <= end and (w - r).days < MAX_AGE]


def label(m, w):
    """Query label: '<Model> noreply' when the name is a prefix of another model released by then."""
    kids = [c for c in released(w) if c.startswith(m + ".")]
    return f"Claude {m} noreply" if kids else f"Claude {m}"


def newest_two(fam, w):
    ms = sorted([m for m in released(w) if m.startswith(fam + " ")], key=lambda m: PRICES[m][0])
    return ms[-2:]


def jobs(plan):
    out = []
    if plan == "first":
        d = dt.date(2025, 10, 27)
        while d <= dt.date(2025, 12, 29):
            if (d - FIRST).days % 28:                    # 4-weekly windows are covered by the other plans
                out += [(d, "Claude"), (d, "Claude Opus"), (d, "Claude Sonnet")]
            d += dt.timedelta(days=7)
        for w in windows("4w"):
            if dt.date(2025, 4, 1) <= w <= dt.date(2025, 10, 31):
                out += [(w, "Claude"), (w, "Claude Sonnet"), (w, "Claude Opus")]
        return out
    for w in sorted(windows(plan), reverse=True):        # most recent first
        if w < NAMES_FROM:
            continue
        rel = released(w)
        fams = [f for f in FAMILIES if any(m.startswith(f + " ") for m in rel)]
        if plan == "price":
            out += [(w, f"Claude {f}") for f in fams if f in ("Haiku", "Fable")]
            for f in fams:
                two = newest_two(f, w)
                if len(two) == 2 and PRICES[two[0]][1:] != PRICES[two[1]][1:]:
                    out += [(w, label(m, w)) for m in reversed(two)]
        elif plan in ("models", "2w"):
            if plan == "2w":
                out += [(w, "Claude")] + [(w, f"Claude {f}") for f in fams]
            for f in fams:
                out += [(w, label(m, w)) for m in reversed(newest_two(f, w))]
        else:
            raise SystemExit(f"unknown plan {plan}")
    return out


def count(q):
    wait = 90
    for _ in range(12):
        r = subprocess.run(["gh", "api", "-X", "GET", "search/commits", "-f", f"q={q}", "-f", "per_page=1"],
                           capture_output=True, text=True)
        try:
            d = json.loads(r.stdout)
            if "total_count" in d:
                return d["total_count"], d.get("incomplete_results")
        except Exception:
            pass
        print(f"  backoff {wait}s: {(r.stdout or r.stderr)[:120]!r}", flush=True)
        time.sleep(wait)
        wait = min(wait * 2, 900)
    return None, None


def main():
    done = set()
    # skip queries already answered here or in the older family / totals files of the same windows
    older = [os.path.join(os.path.dirname(OUT), n) for n in ("github_claude_family.csv", "github_claude_commits.csv")]
    for path in [OUT] + older:
        if os.path.exists(path):
            with open(path) as f:
                for row in csv.DictReader(f):
                    if row["total_count"] not in ("", "None"):
                        done.add((row["week_start"], row["query"]))
    new = not os.path.exists(OUT)
    todo = [(w, q) for w, q in jobs(PLAN) if (str(w), q) not in done]
    print(f"{PLAN}: {len(todo)} queries to run", flush=True)
    with open(OUT, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["week_start", "week_end", "query", "total_count", "incomplete", "collected_at"])
        for i, (w, who) in enumerate(todo):
            end = w + dt.timedelta(days=6)
            phrase = who[:-len(" noreply")] + " noreply" if who.endswith(" noreply") else who
            n, inc = count(f'"Co-Authored-By: {phrase}" committer-date:{w}..{end}')
            wr.writerow([w, end, who, n, inc, dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")])
            f.flush()
            print(f"{i + 1}/{len(todo)} {w} {who}: {n} {inc}", flush=True)
            time.sleep(PACE)
    print("done", flush=True)


if __name__ == "__main__":
    main()
