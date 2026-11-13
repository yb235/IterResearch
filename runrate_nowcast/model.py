"""State-space model for run-rate revenue: three layers in one filter.

State (one step per calendar day):
  Total layer
    l      log instantaneous annualized revenue ($B), on the GAAP revenue basis
    g      log growth per month (local linear trend: l' = l + g/30.44; optionally damped
           toward a long-run rate: g' = g + kappa (g_bar - g)/30.44)
    b      log gap between reported run-rate (company definition) and GAAP revenue
  Segment layer
    s_k    log share of segment k in total revenue (random walk). Partition segments
           (consumption vs subscription) carry a soft sum(exp(s_k)) = 1 constraint.
           Overlays (cloud channel, Claude Code) do not.
  Leading-indicator layer
    a_i    log intercept of indicator i (random walk). The indicator is observed as
           log(I_t) = a_i + beta_i * (l + s_seg)(t - lag_i) + noise.
           The drifting intercept means the indicator-to-revenue relationship is not
           taken as fixed: levels come from audited and official figures, and the
           indicators mostly carry information about changes. A known break in how a
           source counts (catalog `breaks`) adds a one-off variance bump to a_i.
  Indicator families (optional)
    c_f    shared drift of an indicator family (e.g. npm and PyPI SDK downloads) relative
           to revenue. Without it, two indicators drifting together are counted as two
           independent confirmations.
  Growth volatility (optional)
    the growth-shock variance is multiplied by exp(v_q) in calendar quarter q
    (piecewise stochastic volatility; v_q are parameters, see bayes.py).
  Revenue accumulators
    A_k    revenue accumulated so far in period k ($B). Quarterly GAAP revenue is an
           exact integral of the daily path.

Revenue on day t is exp(l)/365, a nonlinear function of the state. The predict step uses
statistical linear regression about a linearization point; about the filter's own
estimate this is exact moment matching (Stein's lemma). The smoother is an iterated
posterior linearization smoother: each pass re-linearizes the dynamics and the share
constraint about the previous pass's smoothed marginals.
Claims like "over $X" or "at most $X" are truncated observations, applied by
moment-matching the truncated normal (assumed-density filtering).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import log_ndtr
from scipy.stats import norm

from . import config

L, G, B = 0, 1, 2
N_BASE = 3
Z80 = 1.2815515655446004
BREAK_SD = 0.5


@dataclass
class Period:
    key: str
    start: pd.Timestamp
    end: pd.Timestamp


def period_key(start, end) -> str:
    return f"{pd.Timestamp(start):%Y-%m-%d}..{pd.Timestamp(end):%Y-%m-%d}"


@dataclass
class Layout:
    segments: list
    indicators: list
    periods: list
    factors: list = field(default_factory=list)

    @property
    def n(self):
        return N_BASE + len(self.segments) + len(self.indicators) + len(self.factors) + len(self.periods)

    def seg(self, name):
        return N_BASE + self.segments.index(name)

    def ind(self, name):
        return N_BASE + len(self.segments) + self.indicators.index(name)

    def fac(self, name):
        return N_BASE + len(self.segments) + len(self.indicators) + self.factors.index(name)

    @property
    def acc0(self):
        return N_BASE + len(self.segments) + len(self.indicators) + len(self.factors)

    @property
    def n_core(self):
        return N_BASE + len(self.segments)


@dataclass
class Obs:
    obs_id: str
    t: int
    metric: str
    kind: str
    y: float
    H: np.ndarray
    offset: float
    base_var: float
    date_sd: float
    tier: str
    constraint: bool = False
    y_hi: float | None = None      # kind "lower" with an upper edge: y <= value <= y_hi (config.LOWER_UPPER_RATIO)


@dataclass
class Innovation:
    obs_id: str
    date: pd.Timestamp
    metric: str
    kind: str
    tier: str
    y: float
    mu: float
    sd: float
    z: float
    p_tail: float      # two-sided p for points; P(constraint holds) for bounds
    loglik: float


@dataclass
class Result:
    start: pd.Timestamp
    dates: pd.DatetimeIndex
    layout: Layout
    m_pred: np.ndarray
    P_pred: np.ndarray
    m_f: np.ndarray
    P_f: np.ndarray
    ms: np.ndarray | None
    Ps: np.ndarray | None
    loglik: float
    innovations: list = field(default_factory=list)
    ms_prev: np.ndarray | None = None
    Ps_prev: np.ndarray | None = None
    iterations: int = 1

    @property
    def periods(self):
        return self.layout.periods

    # helpers
    def idx(self, date) -> int:
        i = (pd.Timestamp(date) - self.start).days
        if not 0 <= i < len(self.dates):
            raise IndexError(f"{date} outside model range {self.dates[0]:%Y-%m-%d}..{self.dates[-1]:%Y-%m-%d}")
        return i

    def state(self, date, smoothed=True):
        i = self.idx(date)
        if smoothed and self.ms is not None:
            return self.ms[i], self.Ps[i]
        return self.m_f[i], self.P_f[i]

    def pk(self, key) -> int:
        for k, p in enumerate(self.layout.periods):
            if p.key == key:
                return self.layout.acc0 + k
        raise KeyError(key)

    def log_runrate(self, date, definition="instant", smoothed=True, segment="total"):
        """Mean and sd of log run-rate (total or one segment) at `date` under a definition."""
        lag = config.DEFINITION_LAG_DAYS[definition]
        d = pd.Timestamp(date) - pd.Timedelta(days=round(lag))
        m, P = self.state(d, smoothed)
        h = np.zeros(len(m))
        h[L] = 1.0
        if definition != "instant":
            h[B] = 1.0
        if segment != "total":
            h[self.layout.seg(segment)] = 1.0
        return float(h @ m), float(math.sqrt(max(h @ P @ h, 0.0)))

    def runrate(self, date, definition="instant", smoothed=True, segment="total") -> dict:
        mu, sd = self.log_runrate(date, definition, smoothed, segment)
        return lognormal_summary(mu, sd)

    def share(self, date, segment, smoothed=True) -> dict:
        m, P = self.state(date, smoothed)
        j = self.layout.seg(segment)
        return lognormal_summary(float(m[j]), float(math.sqrt(max(P[j, j], 0.0))))

    def indicator_intercept(self, date, name, smoothed=True):
        m, P = self.state(date, smoothed)
        j = self.layout.ind(name)
        return float(m[j]), float(math.sqrt(max(P[j, j], 0.0)))

    def growth(self, date, smoothed=True):
        m, P = self.state(date, smoothed)
        return float(m[G]), float(math.sqrt(P[G, G]))

    def gap(self, date, smoothed=True):
        m, P = self.state(date, smoothed)
        return float(m[B]), float(math.sqrt(P[B, B]))

    def period_revenue(self, key, smoothed=True) -> dict:
        j = self.pk(key)
        p = self.layout.periods[j - self.layout.acc0]
        end = min(p.end, self.dates[-1])
        m, P = self.state(end, smoothed)
        mu, sd = float(m[j]), float(math.sqrt(max(P[j, j], 0.0)))
        # Revenue is a sum of lognormals, so it is right-skewed. The quantiles come from
        # a lognormal with the same mean and variance.
        if mu > 0:
            s2 = math.log1p((sd / mu) ** 2)
            lm = math.log(mu) - s2 / 2
            q10, q90 = math.exp(lm - Z80 * math.sqrt(s2)), math.exp(lm + Z80 * math.sqrt(s2))
        else:
            q10, q90 = mu - Z80 * sd, mu + Z80 * sd
        return {"mean": mu, "sd": sd, "q10": q10, "q90": q90, "complete": bool(end >= p.end)}


def lognormal_summary(mu, sd) -> dict:
    return {"median": math.exp(mu), "mean": math.exp(mu + sd * sd / 2),
            "q10": math.exp(mu - Z80 * sd), "q90": math.exp(mu + Z80 * sd),
            "log_mu": mu, "log_sd": sd}


def log_ndtr_diff(b: float, a: float) -> float:
    """log(Phi(b) - Phi(a)) for b > a, accurate when both lie in the same tail."""
    if a > 0:                       # Phi(b) - Phi(a) = Phi(-a) - Phi(-b)
        la, lb = float(log_ndtr(-a)), float(log_ndtr(-b))
        return la + math.log1p(-math.exp(lb - la))
    if b < 0:
        la, lb = float(log_ndtr(a)), float(log_ndtr(b))
        return lb + math.log1p(-math.exp(la - lb))
    return math.log(max(1.0 - math.exp(float(log_ndtr(a))) - math.exp(float(log_ndtr(-b))), 1e-300))


def lower_claim_rule(row, y: float) -> tuple[str, float | None]:
    """(kind, y_hi) of a claim under config.LOWER_UPPER_RATIO (see config.py and lower_rule.py).

    Only kind "lower" claims of run-rate or revenue that are not forecasts change. y is the claim
    as obs_vector returns it: log value for run-rate, $B for revenue.
    """
    rule = config.LOWER_UPPER_RATIO
    if rule is None or row.kind != "lower" or row.metric not in ("run_rate", "revenue") \
            or int(_col(row, "is_forecast", 0)) == 1:
        return row.kind, None
    if rule == "point":
        return "point", None
    r = float(rule)
    if not r > 1.0:
        raise ValueError(f"config.LOWER_UPPER_RATIO must be None, 'point' or a ratio > 1, not {rule!r}")
    return "lower", (y + math.log(r) if row.metric == "run_rate" else y * r)


def _col(row, name, default):
    v = getattr(row, name, default)
    if v is None or (isinstance(v, str) and v == ""):
        return default
    try:
        if pd.isna(v):             # NaN, None, NaT
            return default
    except (TypeError, ValueError):
        pass
    return v


class Model:
    def __init__(self, params: dict | None = None, start: str = config.MODEL_START):
        p = dict(config.DEFAULT_PARAMS)
        tiers = {k: dict(v) for k, v in config.TIERS.items()}
        segments = {k: dict(v) for k, v in config.SEGMENTS.items()}
        indicators = {}
        if params:
            p.update({k: v for k, v in params.items() if k not in ("tiers", "segments", "indicators")})
            for t, v in params.get("tiers", {}).items():
                tiers.setdefault(t, {}).update(v)
            for s, v in params.get("segments", {}).items():
                segments.setdefault(s, {}).update(v)
            indicators = {k: dict(v) for k, v in params.get("indicators", {}).items()}
        p["tiers"], p["segments"], p["indicators"] = tiers, segments, indicators
        self.p = p
        self.start = pd.Timestamp(start)

    # ------------------------------------------------------------------ build
    def layout(self, obs_df, queries, segments=None) -> Layout:
        segs, inds = set(segments or []), []
        if obs_df is not None and len(obs_df):
            for r in obs_df.itertuples():
                s = _col(r, "segment", "total")
                if s != "total":
                    segs.add(s)
                if r.metric == "indicator":
                    name = _col(r, "indicator", "")
                    if name not in inds:
                        inds.append(name)
                    s = self.p["indicators"][name].get("segment", "total")
                    if s != "total":
                        segs.add(s)
        # Any partition segment in use pulls in the rest of the partition.
        if any(self.p["segments"][s]["partition"] for s in segs):
            segs |= {s for s, v in self.p["segments"].items() if v["partition"]}
        order = [s for s in self.p["segments"] if s in segs]
        facs = []
        for name in inds:
            f = self.p["indicators"][name].get("factor") or ""
            if f and f not in facs:
                facs.append(f)
        return Layout(order, inds, self._periods(obs_df, queries), facs)

    def _periods(self, obs_df, queries):
        seen, out = set(), []
        rows = [] if obs_df is None else obs_df[obs_df.metric == "revenue"][["period_start", "period_end"]].values.tolist()
        for s, e in rows + [tuple(q) for q in (queries or [])]:
            s, e = pd.Timestamp(s), pd.Timestamp(e)
            key = period_key(s, e)
            if key in seen:
                continue
            if s < self.start:
                raise ValueError(f"period {key} starts before model start {self.start:%Y-%m-%d}")
            seen.add(key)
            out.append(Period(key, s, e))
        return out

    def obs_vector(self, row, lay: Layout):
        """(H, y, offset, base_var, date_sd) for one claim; shared by the filter and predictive."""
        n = lay.n
        H = np.zeros(n)
        tier = self.p["tiers"][row.tier]
        seg = _col(row, "segment", "total")
        if row.metric == "run_rate":
            H[L] = H[B] = 1.0
            if seg != "total":
                H[lay.seg(seg)] = 1.0
            y, offset = math.log(row.value), tier.get("bias", 0.0)
            base_var, date_sd = tier["log_sd"] ** 2, float(_col(row, "date_sd_days", 0.0))
        elif row.metric == "revenue":
            if seg != "total":
                raise ValueError(f"{row.obs_id}: segment revenue is entered as metric=share")
            H[lay.acc0 + [p.key for p in lay.periods].index(period_key(row.period_start, row.period_end))] = 1.0
            y, offset = float(row.value), 0.0
            base_var, date_sd = (tier["log_sd"] * row.value) ** 2, 0.0
        elif row.metric == "share":
            H[lay.seg(seg)] = 1.0
            y, offset = math.log(row.value), 0.0
            # A period share is entered at the revenue-weighted centre of the period. If the
            # share follows a random walk with daily sd sigma, that substitution has sd
            # sigma * sqrt(days / 12). The noise is derived, not chosen.
            ps, pe = _col(row, "period_start", None), _col(row, "period_end", None)
            days = (pd.Timestamp(pe) - pd.Timestamp(ps)).days + 1 if ps is not None and pe is not None else 0
            approx = float(self.p["segments"][seg]["sigma"]) * math.sqrt(days / 12.0)
            base_var, date_sd = tier["log_sd"] ** 2 + approx ** 2, 0.0
        elif row.metric == "indicator":
            spec = self.p["indicators"][row.indicator]
            beta = float(spec.get("beta", 1.0))
            H[lay.ind(row.indicator)] = 1.0
            f = spec.get("factor") or ""
            if f:
                H[lay.fac(f)] = 1.0          # shared drift of the indicator family vs revenue
            H[L] = beta
            s = spec.get("segment", "total")
            if s != "total":
                H[lay.seg(s)] = beta
            y, offset = math.log(row.value), 0.0
            base_var, date_sd = float(spec.get("obs_sd", 0.1)) ** 2, 0.0
        else:
            raise ValueError(f"unknown metric {row.metric}")
        if int(_col(row, "is_forecast", 0)) == 1:
            h_months = max((pd.Timestamp(row.ref_date) - pd.Timestamp(row.published)).days, 0) / config.DAYS_PER_MONTH
            base_var += (self.p.get("forecast_sd_per_month", config.FORECAST_SD_PER_MONTH) * h_months) ** 2
        return H, y, offset, base_var, date_sd

    def _compile(self, obs_df, lay):
        out = []
        for r in obs_df.itertuples():
            t = (pd.Timestamp(r.t_eff) - self.start).days
            if t < 0:
                continue
            H, y, offset, base_var, date_sd = self.obs_vector(r, lay)
            kind, y_hi = lower_claim_rule(r, y)
            out.append(Obs(r.obs_id, t, r.metric, kind, y, H, offset, base_var, date_sd, r.tier, y_hi=y_hi))
        return out

    # ------------------------------------------------------------------- run
    def run(self, obs_df, end=None, queries=None, smooth=True, max_iter=15, tol=1e-5,
            segments=None) -> Result:
        """Filter (and smooth) all observations.

        smooth=False: one forward pass, moment-matched about the filter's own estimate.
            This is the online filter, and its likelihood is used for parameter fitting.
        smooth=True: iterated posterior linearization smoother (see module docstring).
        segments: extra segments to carry even when no claim mentions them.
        """
        lay = self.layout(obs_df, queries, segments)
        last = [pd.Timestamp(end)] if end is not None else []
        if obs_df is not None and len(obs_df):
            last.append(pd.Timestamp(obs_df.t_eff.max()))
        last += [p.end for p in lay.periods]
        end = max(last)
        T = (end - self.start).days + 1
        dates = pd.date_range(self.start, periods=T, freq="D")
        obs = self._compile(obs_df, lay) if obs_df is not None and len(obs_df) else []
        by_t = {}
        for o in obs:
            by_t.setdefault(o.t, []).append(o)
        ctx = self._context(lay, obs_df)
        # Growth-shock variance multiplier per day (piecewise stochastic volatility by quarter).
        q = dates.to_period("Q").astype(str)
        ctx["gmult"] = np.array([ctx["vol"].get(k, 1.0) for k in q]) if ctx["vol"] else None

        res = self._pass(ctx, T, dates, lay, by_t, None, smooth)
        if smooth:
            it = 1
            for it in range(2, max_iter + 2):
                lin = (res.ms_prev, res.Ps_prev, res.ms, res.Ps)
                new = self._pass(ctx, T, dates, lay, by_t, lin, smooth)
                k = lay.n_core
                delta = float(np.max(np.abs(new.ms[:, :k] - res.ms[:, :k])))
                res = new
                if delta < tol:
                    break
            res.iterations = it
        return res

    def _context(self, lay: Layout, obs_df):
        p, n = self.p, lay.n
        Q = np.zeros((n, n))
        Q[L, L], Q[G, G], Q[B, B] = p["sigma_l"] ** 2, p["sigma_g"] ** 2, p["sigma_b"] ** 2
        m0 = np.zeros(n)
        m0[L], m0[G], m0[B] = p["prior_l"], p["prior_g"], p["prior_b"]
        P0 = np.zeros((n, n))
        P0[L, L], P0[G, G], P0[B, B] = p["prior_l_sd"] ** 2, p["prior_g_sd"] ** 2, p["prior_b_sd"] ** 2
        for s in lay.segments:
            j, spec = lay.seg(s), p["segments"][s]
            m0[j], P0[j, j], Q[j, j] = math.log(spec["prior_share"]), spec["prior_sd"] ** 2, spec["sigma"] ** 2
        breaks = {}
        for name in lay.indicators:
            j, spec = lay.ind(name), p["indicators"][name]
            # Diffuse intercept centred on the first reading, so the level is learned from data.
            rows = obs_df[(obs_df.metric == "indicator") & (obs_df.indicator == name)]
            first = math.log(float(rows.sort_values("t_eff").value.iloc[0])) if len(rows) else 0.0
            m0[j] = first - float(spec.get("beta", 1.0)) * 3.0
            P0[j, j] = float(spec.get("prior_sd", 5.0)) ** 2
            Q[j, j] = float(spec.get("drift", 0.003)) ** 2
            for d in spec.get("breaks", []) or []:
                breaks.setdefault((pd.Timestamp(d) - self.start).days, []).append(j)
        for f in lay.factors:
            j, spec = lay.fac(f), p.get("factors", {}).get(f, {})
            # The level is carried by the idiosyncratic intercepts; the factor starts at 0
            # and carries only the common drift.
            P0[j, j] = float(spec.get("prior_sd", 0.01)) ** 2
            Q[j, j] = float(spec.get("drift", 0.01)) ** 2
        starts = np.array([(pp.start - self.start).days for pp in lay.periods], dtype=int)
        ends = np.array([(pp.end - self.start).days for pp in lay.periods], dtype=int)
        c = np.zeros(n)
        c[L], c[G] = 1.0, 0.5 / config.DAYS_PER_MONTH   # u = log revenue rate at mid-day
        kappa = float(p.get("growth_reversion", 0.0))
        a_g = np.zeros(n)
        a_g[G] = kappa * float(p.get("growth_longrun", 0.0)) / config.DAYS_PER_MONTH
        part = [lay.seg(s) for s in lay.segments if p["segments"][s]["partition"]]
        return {"Q": Q, "m0": m0, "P0": P0, "starts": starts, "ends": ends, "c": c, "n": n,
                "kappa": kappa, "a_g": a_g, "acc0": lay.acc0, "partition": part, "breaks": breaks,
                "vol": {k: math.exp(float(v)) for k, v in (p.get("vol_scales") or {}).items()}}

    def _transition(self, ctx, t):
        """Linear part M and revenue loading d for the step into day t."""
        n = ctx["n"]
        M = np.eye(n)
        M[L, G] = 1.0 / config.DAYS_PER_MONTH
        M[G, G] = 1.0 - ctx["kappa"] / config.DAYS_PER_MONTH   # damped trend (kappa=0: none)
        d = np.zeros(n)
        k = len(ctx["starts"])
        if k:
            idx = np.arange(k) + ctx["acc0"]
            reset = ctx["starts"] == t
            active = (ctx["starts"] <= t) & (t <= ctx["ends"])
            M[idx[reset], idx[reset]] = 0.0
            d[idx[active]] = 1.0 / config.DAYS_PER_YEAR
        return M, d

    def _constraint_obs(self, ctx, t, mh, Ph):
        """Soft constraint sum_k exp(s_k) = 1 over the partition, linearized about (mh, Ph)."""
        part = ctx["partition"]
        E = np.array([math.exp(mh[j] + 0.5 * Ph[j, j]) for j in part])
        H = np.zeros(ctx["n"])
        H[part] = E
        hbar = float(E.sum())
        Pk = Ph[np.ix_(part, part)]
        var_h = float(np.sum(np.outer(E, E) * np.expm1(Pk)))
        omega = max(var_h - float(E @ Pk @ E), 0.0)
        y = 1.0 - hbar + float(H @ mh)
        return Obs("constraint", t, "constraint", "point", y, H, 0.0,
                   config.CONSTRAINT_SD ** 2 + omega, 0.0, "gaap", constraint=True)

    def _pass(self, ctx, T, dates, lay, by_t, lin, smooth):
        n, c, Q = ctx["n"], ctx["c"], ctx["Q"]
        m, P = ctx["m0"].copy(), ctx["P0"].copy()
        m_pred = np.zeros((T, n)); P_pred = np.zeros((T, n, n))
        m_f = np.zeros((T, n)); P_f = np.zeros((T, n, n))
        Fs = np.zeros((T, n, n)) if smooth else None
        loglik = 0.0
        innovations = []
        for t in range(T):
            M, d = self._transition(ctx, t)
            # Statistical linear regression of f(x) = M x + d e^{c'x} about (mh, Ph):
            #   F = M + E d c',  a = d E (1 - c'mh),  Omega = d d' E^2 (e^v - 1 - v)
            if lin is None:
                mh, Ph = m, P
            elif t == 0:
                mh, Ph = lin[0], lin[1]
            else:
                mh, Ph = lin[2][t - 1], lin[3][t - 1]
            v = max(float(c @ Ph @ c), 0.0)
            E = math.exp(float(c @ mh) + 0.5 * v)
            F = M + E * np.outer(d, c)
            a = d * E * (1.0 - float(c @ mh)) + ctx["a_g"]
            omega = E * E * (math.expm1(v) - v)
            m = F @ m + a
            P = F @ P @ F.T + np.outer(d, d) * omega + Q
            if ctx["gmult"] is not None and ctx["gmult"][t] != 1.0:
                P[G, G] += Q[G, G] * (ctx["gmult"][t] - 1.0)
            for j in ctx["breaks"].get(t, ()):
                P[j, j] += BREAK_SD ** 2          # source changed how it counts
            P = 0.5 * (P + P.T)
            if smooth:
                Fs[t] = F
            m_pred[t], P_pred[t] = m, P
            obs_t = list(by_t.get(t, []))
            if len(ctx["partition"]) >= 2 and t % 7 == 0:
                cm, cP = (m, P) if lin is None else (lin[2][t], lin[3][t])
                obs_t.append(self._constraint_obs(ctx, t, cm, cP))
            for o in obs_t:
                m, P, inn = self._update(m, P, o, dates[t])
                if not o.constraint:
                    loglik += inn.loglik
                    innovations.append(inn)
            m_f[t], P_f[t] = m, P

        res = Result(self.start, dates, lay, m_pred, P_pred, m_f, P_f, None, None, loglik, innovations)
        if not smooth:
            return res
        ms, Ps = m_f.copy(), P_f.copy()
        for t in range(T - 2, -1, -1):
            Pp = P_pred[t + 1]
            Gt = P_f[t] @ Fs[t + 1].T @ np.linalg.pinv(Pp, rcond=1e-10, hermitian=True)
            ms[t] = m_f[t] + Gt @ (ms[t + 1] - m_pred[t + 1])
            Ps_t = P_f[t] + Gt @ (Ps[t + 1] - Pp) @ Gt.T
            Ps[t] = 0.5 * (Ps_t + Ps_t.T)
        Pp = P_pred[0]
        G0 = ctx["P0"] @ Fs[0].T @ np.linalg.pinv(Pp, rcond=1e-10, hermitian=True)
        res.ms_prev = ctx["m0"] + G0 @ (ms[0] - m_pred[0])
        res.Ps_prev = ctx["P0"] + G0 @ (Ps[0] - Pp) @ G0.T
        res.ms, res.Ps = ms, Ps
        return res

    def _update(self, m, P, o: Obs, date):
        D = config.DAYS_PER_MONTH
        H = o.H
        mu = float(H @ m) + o.offset
        R = o.base_var + (m[G] / D * o.date_sd) ** 2
        PH = P @ H
        S = float(H @ PH) + R
        s = math.sqrt(S)
        K = PH / S
        if o.kind == "point":
            r = o.y - mu
            m = m + K * r
            P = P - np.outer(K, PH)
            z = r / s
            ll = float(norm.logpdf(o.y, mu, s))
            p_tail = float(2 * norm.sf(abs(z)))
        else:
            a = (o.y - mu) / s
            if o.kind == "lower" and o.y_hi is not None:     # y <= true value <= y_hi
                b = (o.y_hi - mu) / s
                logZ = log_ndtr_diff(b, a)
                pa = math.exp(norm.logpdf(a) - logZ)
                pb = math.exp(norm.logpdf(b) - logZ)
                Ez = mu + s * (pa - pb)
                Vz = S * min(max(1.0 + a * pa - b * pb - (pa - pb) ** 2, 1e-9), 1.0)
            elif o.kind == "lower":         # true value > y
                logZ = float(log_ndtr(-a))
                lam = math.exp(norm.logpdf(a) - logZ)
                Ez = mu + s * lam
                Vz = S * max(1.0 + a * lam - lam * lam, 1e-9)
            else:                           # true value < y
                logZ = float(log_ndtr(a))
                lam = math.exp(norm.logpdf(a) - logZ)
                Ez = mu - s * lam
                Vz = S * max(1.0 - a * lam - lam * lam, 1e-9)
            m = m + K * (Ez - mu)
            P = P - np.outer(K, PH) + np.outer(K, K) * Vz
            z = a
            ll = logZ
            p_tail = math.exp(logZ)
        P = 0.5 * (P + P.T)
        inn = Innovation(o.obs_id, date, o.metric, o.kind, o.tier, o.y, mu, s, z, p_tail, ll)
        return m, P, inn


def observation_predictive(res: Result, row, model: Model) -> tuple[float, float]:
    """Predictive mean and sd of a claim from the smoothed state.

    The result is in log units for run-rate, share and indicator claims, and in $B for revenue.
    """
    if row.metric == "revenue":
        r = res.period_revenue(period_key(row.period_start, row.period_end))
        _, _, _, base_var, _ = model.obs_vector(row, res.layout)
        # Revenue noise is proportional to the reported value; scale it by the forecast instead,
        # so the predictive sd never depends on the (unknown) value being forecast.
        base_var *= (r["mean"] / float(row.value)) ** 2
        return r["mean"], math.sqrt(r["sd"] ** 2 + base_var)
    H, _, offset, base_var, date_sd = model.obs_vector(row, res.layout)
    m, P = res.state(row.t_eff)
    mu = float(H @ m) + offset
    var = float(H @ P @ H) + base_var + (m[G] / config.DAYS_PER_MONTH * date_sd) ** 2
    return mu, math.sqrt(var)
