"""State-space model for run-rate revenue: filter, truncated updates, RTS smoother.

State (one step per calendar day):
    l    log instantaneous annualized revenue ($B), on the GAAP revenue basis
    g    log growth per month (local linear trend: l' = l + g/30.44; optionally damped
         toward a long-run rate: g' = g + kappa (g_bar - g)/30.44)
    b    log gap between reported run-rate (company definition) and GAAP revenue
         (gross vs net cloud revenue, subscription annualization, and similar)
    A_k  revenue accumulated so far in period k ($B). One accumulator per
         revenue period that is observed or asked about (quarter, year, month).

Revenue on day t is exp(l)/365, so the accumulators are a nonlinear function of l. The
predict step matches moments exactly: if x ~ N(m, P) and u = c'x, then
E[e^u] = exp(c'm + c'Pc/2) and, by Stein's lemma, Cov(x, e^u) = P c E[e^u]. This
gives the mean and covariance of the next state without linearizing.
Observation equations are linear in the state:
    run-rate claim (log):  y = l(t - lag) + b + tier_bias + noise
    revenue claim ($B):    y = A_k(period_end) + noise
Claims like "over $X" or "at most $X" are truncated observations. They are applied by
moment-matching the truncated normal of the observed linear combination, as in
assumed-density filtering.
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


@dataclass
class Period:
    key: str
    start: pd.Timestamp
    end: pd.Timestamp


def period_key(start, end) -> str:
    return f"{pd.Timestamp(start):%Y-%m-%d}..{pd.Timestamp(end):%Y-%m-%d}"


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
    periods: list
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
        for k, p in enumerate(self.periods):
            if p.key == key:
                return N_BASE + k
        raise KeyError(key)

    def log_runrate(self, date, definition="instant", smoothed=True):
        """Mean and sd of log run-rate at `date` under a reporting definition."""
        lag = config.DEFINITION_LAG_DAYS[definition]
        d = pd.Timestamp(date) - pd.Timedelta(days=round(lag))
        m, P = self.state(d, smoothed)
        h = np.zeros(len(m))
        h[L] = 1.0
        if definition != "instant":
            h[B] = 1.0
        return float(h @ m), float(math.sqrt(max(h @ P @ h, 0.0)))

    def runrate(self, date, definition="instant", smoothed=True) -> dict:
        mu, sd = self.log_runrate(date, definition, smoothed)
        return lognormal_summary(mu, sd)

    def growth(self, date, smoothed=True):
        m, P = self.state(date, smoothed)
        return float(m[G]), float(math.sqrt(P[G, G]))

    def gap(self, date, smoothed=True):
        m, P = self.state(date, smoothed)
        return float(m[B]), float(math.sqrt(P[B, B]))

    def period_revenue(self, key, smoothed=True) -> dict:
        j = self.pk(key)
        p = self.periods[j - N_BASE]
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


class Model:
    def __init__(self, params: dict | None = None, start: str = config.MODEL_START):
        p = dict(config.DEFAULT_PARAMS)
        if params:
            p.update({k: v for k, v in params.items() if k != "tiers"})
            if "tiers" in params:
                tiers = {k: dict(v) for k, v in config.TIERS.items()}
                for t, v in params["tiers"].items():
                    tiers.setdefault(t, {}).update(v)
                p["tiers"] = tiers
        self.p = p
        self.start = pd.Timestamp(start)

    # ------------------------------------------------------------------ build
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

    def _compile(self, obs_df, periods, n):
        tiers = self.p["tiers"]
        keyidx = {p.key: N_BASE + k for k, p in enumerate(periods)}
        out = []
        for r in obs_df.itertuples():
            t = (pd.Timestamp(r.t_eff) - self.start).days
            if t < 0:
                continue
            H = np.zeros(n)
            tier = tiers[r.tier]
            if r.metric == "run_rate":
                H[L] = 1.0
                H[B] = 1.0
                y = math.log(r.value)
                offset = tier.get("bias", 0.0)
                base_var = tier["log_sd"] ** 2
                date_sd = float(r.date_sd_days)
            else:
                H[keyidx[period_key(r.period_start, r.period_end)]] = 1.0
                y = float(r.value)
                offset = 0.0
                base_var = (tier["log_sd"] * r.value) ** 2
                date_sd = 0.0
            out.append(Obs(r.obs_id, t, r.metric, r.kind, y, H, offset, base_var, date_sd, r.tier))
        return out

    # ------------------------------------------------------------------- run
    def run(self, obs_df, end=None, queries=None, smooth=True, max_iter=12, tol=1e-6) -> Result:
        """Filter (and smooth) all observations.

        smooth=False: one forward pass. The dynamics are moment-matched around the
            filter's own estimate (assumed-density filter). This is the online filter.
        smooth=True: iterated posterior linearization smoother. Each pass linearizes
            the dynamics statistically around the previous pass's smoothed marginals,
            then reruns the Kalman filter and RTS smoother. Without this, one large
            surprise in a quarterly revenue figure leaves the daily path and the
            accumulated revenue inconsistent.
        """
        periods = self._periods(obs_df, queries)
        n = N_BASE + len(periods)
        last = [pd.Timestamp(end)] if end is not None else []
        if obs_df is not None and len(obs_df):
            last.append(pd.Timestamp(obs_df.t_eff.max()))
        last += [p.end for p in periods]
        end = max(last)
        T = (end - self.start).days + 1
        dates = pd.date_range(self.start, periods=T, freq="D")
        obs = self._compile(obs_df, periods, n) if obs_df is not None else []
        by_t = {}
        for o in obs:
            by_t.setdefault(o.t, []).append(o)
        ctx = self._context(periods, n)

        res = self._pass(ctx, T, dates, periods, by_t, None, smooth)
        if smooth:
            it = 1
            for it in range(2, max_iter + 2):
                lin = (res.ms_prev, res.Ps_prev, res.ms, res.Ps)
                new = self._pass(ctx, T, dates, periods, by_t, lin, smooth)
                delta = float(np.max(np.abs(new.ms[:, :N_BASE] - res.ms[:, :N_BASE])))
                res = new
                if delta < tol:
                    break
            res.iterations = it
        return res

    def _context(self, periods, n):
        p = self.p
        Q = np.zeros((n, n))
        Q[L, L], Q[G, G], Q[B, B] = p["sigma_l"] ** 2, p["sigma_g"] ** 2, p["sigma_b"] ** 2
        m0 = np.zeros(n)
        m0[L], m0[G], m0[B] = p["prior_l"], p["prior_g"], p["prior_b"]
        P0 = np.zeros((n, n))
        P0[L, L], P0[G, G], P0[B, B] = p["prior_l_sd"] ** 2, p["prior_g_sd"] ** 2, p["prior_b_sd"] ** 2
        starts = np.array([(pp.start - self.start).days for pp in periods], dtype=int)
        ends = np.array([(pp.end - self.start).days for pp in periods], dtype=int)
        c = np.zeros(n)
        c[L], c[G] = 1.0, 0.5 / config.DAYS_PER_MONTH   # u = log revenue rate at mid-day
        # Damped trend: g' = g + kappa (g_bar - g) / 30.44, so growth reverts to g_bar
        # with a half-life of ln2/kappa months.
        kappa = float(p.get("growth_reversion", 0.0))
        a_g = np.zeros(n)
        a_g[G] = kappa * float(p.get("growth_longrun", 0.0)) / config.DAYS_PER_MONTH
        return {"Q": Q, "m0": m0, "P0": P0, "starts": starts, "ends": ends, "c": c, "n": n,
                "kappa": kappa, "a_g": a_g}

    def _transition(self, ctx, t):
        """Linear part M and revenue loading d for the step into day t."""
        n = ctx["n"]
        M = np.eye(n)
        M[L, G] = 1.0 / config.DAYS_PER_MONTH
        M[G, G] = 1.0 - ctx["kappa"] / config.DAYS_PER_MONTH   # damped trend (kappa=0: none)
        d = np.zeros(n)
        k = len(ctx["starts"])
        if k:
            idx = np.arange(k) + N_BASE
            reset = ctx["starts"] == t
            active = (ctx["starts"] <= t) & (t <= ctx["ends"])
            M[idx[reset], idx[reset]] = 0.0
            d[idx[active]] = 1.0 / config.DAYS_PER_YEAR
        return M, d

    def _pass(self, ctx, T, dates, periods, by_t, lin, smooth):
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
            # About the filter's own (m, P), this is exact moment matching.
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
            P = 0.5 * (P + P.T)
            if smooth:
                Fs[t] = F
            m_pred[t], P_pred[t] = m, P
            for o in by_t.get(t, []):
                m, P, inn = self._update(m, P, o, dates[t])
                loglik += inn.loglik
                innovations.append(inn)
            m_f[t], P_f[t] = m, P

        res = Result(self.start, dates, periods, m_pred, P_pred, m_f, P_f, None, None, loglik, innovations)
        if not smooth:
            return res
        ms, Ps = m_f.copy(), P_f.copy()
        for t in range(T - 2, -1, -1):
            Pp = P_pred[t + 1]
            Gt = P_f[t] @ Fs[t + 1].T @ np.linalg.pinv(Pp, rcond=1e-10, hermitian=True)
            ms[t] = m_f[t] + Gt @ (ms[t + 1] - m_pred[t + 1])
            Ps_t = P_f[t] + Gt @ (Ps[t + 1] - Pp) @ Gt.T
            Ps[t] = 0.5 * (Ps_t + Ps_t.T)
        # smoothed prior (state before day 0), used to linearize the first step
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
            if o.kind == "lower":           # true value > y
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


def observation_predictive(res: Result, row, tiers) -> tuple[float, float]:
    """Predictive mean and sd of a claim (log for run-rate, $B for revenue), from smoothed state."""
    D = config.DAYS_PER_MONTH
    tier = tiers[row.tier]
    if row.metric == "run_rate":
        m, P = res.state(row.t_eff)
        h = np.zeros(len(m)); h[L] = 1; h[B] = 1
        mu = float(h @ m) + tier.get("bias", 0.0)
        var = float(h @ P @ h) + tier["log_sd"] ** 2 + (m[G] / D * row.date_sd_days) ** 2
        return mu, math.sqrt(var)
    key = period_key(row.period_start, row.period_end)
    r = res.period_revenue(key)
    var = r["sd"] ** 2 + (tier["log_sd"] * row.value) ** 2
    return r["mean"], math.sqrt(var)
