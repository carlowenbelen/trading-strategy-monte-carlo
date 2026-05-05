"""
🎲  Trading Strategy Monte Carlo Simulator
==========================================
Stress-test a trading strategy by simulating thousands of possible equity curves.
Given a strategy's win rate, risk per trade, and reward-to-risk ratio, this tool
runs N independent simulations of M trades each and reports the distribution of
outcomes — including probability of ruin, probability of doubling capital, and
the worst drawdown you should expect.

Why this exists
---------------
A single backtest is one path through randomness. Two strategies with identical
expected value can have wildly different real-world outcomes due to luck and
sequencing. Monte Carlo simulation gives you the *distribution* of outcomes,
not just one sample — which is what professional risk managers actually use.

Quick start
-----------
    pip install -r requirements.txt
    python monte_carlo.py --preset goshenfx-inspired
    python monte_carlo.py --win-rate 0.45 --risk 0.01 --rr 2.0 --trades 200 --runs 10000
    python monte_carlo.py --list-presets

Output
------
A self-contained HTML report (`reports/<timestamp>.html`) with 4 charts:
  1. Fan chart — every simulated equity curve overlaid
  2. Histogram of final equity (return distribution)
  3. Histogram of max drawdown
  4. Probability gauges — P(ruin), P(2x), P(5x)

Author
------
Carl Owen E. Belen — https://github.com/YOUR-USERNAME
Built as a companion analysis tool for my GoshenFX trading bots.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ============================================================
# 📐  Strategy definition
# ============================================================
@dataclass
class Strategy:
    """A trading strategy described in risk-and-edge terms."""
    name: str
    win_rate: float            # 0–1, probability of a winning trade
    risk_per_trade: float      # 0–1, fraction of current equity risked per trade
    reward_to_risk: float      # >0, win = risk × this, loss = -risk
    n_trades: int              # number of trades to simulate per run
    description: str = ""

    @property
    def expected_value(self) -> float:
        """Expected return per trade as fraction of current equity."""
        return (self.win_rate * self.risk_per_trade * self.reward_to_risk
                - (1 - self.win_rate) * self.risk_per_trade)

    @property
    def edge_per_trade(self) -> float:
        """Same as expected_value, exposed under a friendlier name."""
        return self.expected_value


# ============================================================
# 🎲  Core simulator (numpy-vectorized — fast)
# ============================================================
def simulate(
    strategy: Strategy,
    n_runs: int = 10_000,
    starting_capital: float = 10_000.0,
    seed: int | None = None,
) -> dict:
    """Run N independent Monte Carlo paths.

    Returns a dict containing:
        equity:         (n_runs, n_trades+1) array of equity curves
        final_equity:   (n_runs,) array
        max_drawdown:   (n_runs,) array — peak-to-trough as fraction
        params:         the input strategy + capital
    """
    rng = np.random.default_rng(seed)
    n_trades = strategy.n_trades

    # 1. Generate trade outcomes: True = win, False = loss
    wins = rng.random((n_runs, n_trades)) < strategy.win_rate

    # 2. Convert to per-trade fractional return
    win_return = strategy.risk_per_trade * strategy.reward_to_risk
    loss_return = -strategy.risk_per_trade
    returns = np.where(wins, win_return, loss_return)

    # 3. Compound: equity[i, t] = capital × Π(1 + returns[i, :t+1])
    equity = starting_capital * np.cumprod(1.0 + returns, axis=1)

    # 4. Prepend starting capital column
    start_col = np.full((n_runs, 1), starting_capital)
    equity = np.concatenate([start_col, equity], axis=1)

    # 5. Max drawdown per run
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdown = (running_max - equity) / running_max
    max_drawdown = drawdown.max(axis=1)

    return {
        "equity": equity,
        "final_equity": equity[:, -1],
        "max_drawdown": max_drawdown,
        "params": {
            "strategy": asdict(strategy),
            "starting_capital": starting_capital,
            "n_runs": n_runs,
            "edge_per_trade": strategy.edge_per_trade,
        },
    }


# ============================================================
# 📊  Statistics
# ============================================================
def summarize(result: dict, ruin_threshold: float = 0.5) -> dict:
    """Compute the headline statistics from a simulation result.

    ruin_threshold: equity drops to this fraction of starting capital → 'ruined'.
    """
    final = result["final_equity"]
    dd = result["max_drawdown"]
    starting = result["params"]["starting_capital"]
    n_runs = len(final)

    return_pct = (final / starting - 1.0) * 100

    return {
        "n_runs": n_runs,
        "median_return_pct": float(np.median(return_pct)),
        "mean_return_pct": float(np.mean(return_pct)),
        "p5_return_pct": float(np.percentile(return_pct, 5)),
        "p95_return_pct": float(np.percentile(return_pct, 95)),
        "median_max_drawdown_pct": float(np.median(dd) * 100),
        "p95_max_drawdown_pct": float(np.percentile(dd, 95) * 100),
        "prob_ruin": float(np.mean(final < starting * ruin_threshold)),
        "prob_2x": float(np.mean(final >= starting * 2)),
        "prob_5x": float(np.mean(final >= starting * 5)),
        "prob_10x": float(np.mean(final >= starting * 10)),
        "prob_loss": float(np.mean(final < starting)),
        "ruin_threshold_pct": ruin_threshold * 100,
    }


# ============================================================
# 🎨  Charts (matplotlib → base64 PNG)
# ============================================================
def _matplotlib():
    """Lazy-import matplotlib so we don't pay the cost when listing presets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _fig_to_b64(fig) -> str:
    """Save a matplotlib figure to a base64 PNG string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def chart_fan(result: dict) -> str:
    plt = _matplotlib()
    equity = result["equity"]
    starting = result["params"]["starting_capital"]
    n_runs, _ = equity.shape

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#0f0f10")
    ax.set_facecolor("#0f0f10")

    # Plot up to 800 random sample paths in light gray
    sample = np.random.choice(n_runs, size=min(800, n_runs), replace=False)
    for i in sample:
        ax.plot(equity[i], color="#64a3df", alpha=0.04, linewidth=0.6)

    # Percentile bands
    p5 = np.percentile(equity, 5, axis=0)
    p50 = np.percentile(equity, 50, axis=0)
    p95 = np.percentile(equity, 95, axis=0)
    ax.plot(p50, color="#a3e635", linewidth=2.2, label="Median path")
    ax.plot(p5, color="#fb7185", linewidth=1.4, linestyle="--", label="5th percentile (worst-case)")
    ax.plot(p95, color="#22d3ee", linewidth=1.4, linestyle="--", label="95th percentile (best-case)")
    ax.axhline(starting, color="#888", linewidth=0.8, linestyle=":", label="Starting capital")

    ax.set_title(f"{n_runs:,} Monte Carlo equity paths", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Trade #", color="#aaa")
    ax.set_ylabel("Equity ($)", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, alpha=0.1)
    ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white", loc="upper left")

    out = _fig_to_b64(fig)
    plt.close(fig)
    return out


def chart_final_distribution(result: dict) -> str:
    plt = _matplotlib()
    final = result["final_equity"]
    starting = result["params"]["starting_capital"]

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0f0f10")
    ax.set_facecolor("#0f0f10")

    return_pct = (final / starting - 1.0) * 100
    ax.hist(return_pct, bins=80, color="#22d3ee", alpha=0.85, edgecolor="none")
    ax.axvline(0, color="#888", linewidth=1, linestyle=":", label="Break-even")
    ax.axvline(np.median(return_pct), color="#a3e635", linewidth=2,
               label=f"Median: {np.median(return_pct):+.0f}%")

    ax.set_title("Distribution of final returns", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Final return (%)", color="#aaa")
    ax.set_ylabel("Number of paths", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, alpha=0.1, axis="y")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")

    out = _fig_to_b64(fig)
    plt.close(fig)
    return out


def chart_drawdown(result: dict) -> str:
    plt = _matplotlib()
    dd = result["max_drawdown"] * 100  # to %

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0f0f10")
    ax.set_facecolor("#0f0f10")

    ax.hist(dd, bins=80, color="#fb7185", alpha=0.85, edgecolor="none")
    ax.axvline(np.median(dd), color="#a3e635", linewidth=2,
               label=f"Median: {np.median(dd):.0f}%")
    ax.axvline(np.percentile(dd, 95), color="#fbbf24", linewidth=2, linestyle="--",
               label=f"95th percentile: {np.percentile(dd, 95):.0f}%")

    ax.set_title("Distribution of max drawdown", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Max drawdown (%)", color="#aaa")
    ax.set_ylabel("Number of paths", color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, alpha=0.1, axis="y")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333", labelcolor="white")

    out = _fig_to_b64(fig)
    plt.close(fig)
    return out


# ============================================================
# 📄  HTML report
# ============================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Monte Carlo Report — {strategy_name}</title>
<style>
:root {{ --bg: #0f0f10; --card: #1a1a1a; --text: #f0f0f0; --muted: #888; --accent: #a3e635; --warn: #fb7185; }}
body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin: 0; padding: 0; line-height: 1.6; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 3rem 2rem 5rem; }}
h1 {{ font-size: 2.5rem; margin: 0 0 0.4rem; letter-spacing: -0.02em; }}
.subtitle {{ color: var(--muted); margin-bottom: 2.5rem; }}
.eyebrow {{ color: var(--accent); text-transform: uppercase; letter-spacing: 2px; font-size: 0.78rem; font-weight: 700; margin-bottom: 0.4rem; }}
.params {{ background: var(--card); border: 1px solid #222; border-radius: 12px; padding: 1.5rem 1.75rem; margin-bottom: 2rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; }}
.param-label {{ font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 0.2rem; }}
.param-value {{ font-size: 1.4rem; font-weight: 700; color: white; }}
.stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2.5rem; }}
.stat {{ background: var(--card); border: 1px solid #222; border-radius: 12px; padding: 1.25rem 1.5rem; }}
.stat-label {{ color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; }}
.stat-value {{ font-size: 2rem; font-weight: 700; line-height: 1; color: white; }}
.stat.warn .stat-value {{ color: var(--warn); }}
.stat.good .stat-value {{ color: var(--accent); }}
.chart {{ background: var(--card); border: 1px solid #222; border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem; }}
.chart img {{ width: 100%; display: block; border-radius: 8px; }}
.chart-title {{ color: var(--muted); font-size: 0.9rem; padding: 0.6rem 0.4rem; }}
footer {{ color: var(--muted); border-top: 1px solid #222; padding-top: 1.5rem; margin-top: 3rem; font-size: 0.85rem; }}
</style>
</head>
<body>
<div class="container">
  <div class="eyebrow">Monte Carlo Report</div>
  <h1>{strategy_name}</h1>
  <div class="subtitle">{strategy_description}<br>{n_runs:,} runs &middot; {n_trades} trades each &middot; generated {generated_at}</div>

  <div class="params">
    <div><div class="param-label">Win rate</div><div class="param-value">{win_rate:.0%}</div></div>
    <div><div class="param-label">Risk / trade</div><div class="param-value">{risk:.1%}</div></div>
    <div><div class="param-label">Reward : risk</div><div class="param-value">{rr:.1f}×</div></div>
    <div><div class="param-label">Edge / trade</div><div class="param-value">{edge:+.2%}</div></div>
    <div><div class="param-label">Starting capital</div><div class="param-value">${starting:,.0f}</div></div>
  </div>

  <div class="stat-grid">
    <div class="stat"><div class="stat-label">Median return</div><div class="stat-value">{median_return:+.0f}%</div></div>
    <div class="stat {ruin_class}"><div class="stat-label">Probability of ruin</div><div class="stat-value">{prob_ruin:.1%}</div></div>
    <div class="stat good"><div class="stat-label">P(2× capital)</div><div class="stat-value">{prob_2x:.1%}</div></div>
    <div class="stat"><div class="stat-label">P(5× capital)</div><div class="stat-value">{prob_5x:.1%}</div></div>
    <div class="stat"><div class="stat-label">Median max drawdown</div><div class="stat-value">{med_dd:.0f}%</div></div>
    <div class="stat warn"><div class="stat-label">Worst-case drawdown (P95)</div><div class="stat-value">{p95_dd:.0f}%</div></div>
  </div>

  <div class="chart"><img src="data:image/png;base64,{chart_fan}"><div class="chart-title">Fan chart — all simulated equity paths</div></div>
  <div class="chart"><img src="data:image/png;base64,{chart_final}"><div class="chart-title">Histogram of final returns</div></div>
  <div class="chart"><img src="data:image/png;base64,{chart_dd}"><div class="chart-title">Histogram of max drawdown</div></div>

  <footer>
    Generated by <strong>Trading Strategy Monte Carlo Simulator</strong>.<br>
    Ruin defined as ending equity below {ruin_threshold:.0f}% of starting capital.
  </footer>
</div>
</body>
</html>
"""


def render_report(result: dict, summary: dict, output_path: str | Path):
    """Generate a self-contained HTML report at output_path."""
    s = result["params"]["strategy"]
    starting = result["params"]["starting_capital"]
    edge = result["params"]["edge_per_trade"]
    ruin_class = "warn" if summary["prob_ruin"] >= 0.05 else "good"

    print("  → rendering charts …")
    fan = chart_fan(result)
    final = chart_final_distribution(result)
    dd = chart_drawdown(result)

    html = HTML_TEMPLATE.format(
        strategy_name=s["name"],
        strategy_description=s.get("description", "") or "Custom strategy",
        n_runs=summary["n_runs"],
        n_trades=s["n_trades"],
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        win_rate=s["win_rate"],
        risk=s["risk_per_trade"],
        rr=s["reward_to_risk"],
        edge=edge,
        starting=starting,
        median_return=summary["median_return_pct"],
        prob_ruin=summary["prob_ruin"],
        prob_2x=summary["prob_2x"],
        prob_5x=summary["prob_5x"],
        med_dd=summary["median_max_drawdown_pct"],
        p95_dd=summary["p95_max_drawdown_pct"],
        ruin_threshold=summary["ruin_threshold_pct"],
        ruin_class=ruin_class,
        chart_fan=fan,
        chart_final=final,
        chart_dd=dd,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ============================================================
# 🎯  Presets (loaded from presets.json)
# ============================================================
PRESETS_FILE = Path(__file__).resolve().parent / "presets.json"


def load_presets() -> dict[str, Strategy]:
    if not PRESETS_FILE.exists():
        return {}
    data = json.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    return {key: Strategy(**val) for key, val in data.items()}


# ============================================================
# 🖨️  Terminal output
# ============================================================
class K:
    CY = "\033[96m"; G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
    M = "\033[35m"; B = "\033[94m"; W = "\033[97m"; DIM = "\033[2m"
    BOLD = "\033[1m"; END = "\033[0m"


def c(t, color):
    return f"{color}{t}{K.END}"


def print_summary(strategy: Strategy, summary: dict, starting: float):
    print()
    print(c("═" * 68, K.DIM))
    print(c(f"  🎲  Monte Carlo Result — {strategy.name}", K.BOLD + K.CY))
    print(c("═" * 68, K.DIM))
    print()
    print(f"  Runs:               {summary['n_runs']:>12,}")
    print(f"  Trades / run:       {strategy.n_trades:>12,}")
    print(f"  Edge per trade:     {strategy.edge_per_trade:>11.2%}")
    print(f"  Starting capital:   {('$' + f'{starting:,.0f}'):>12}")
    print()
    g = K.G if summary["median_return_pct"] >= 0 else K.R
    r_color = K.R if summary["prob_ruin"] >= 0.05 else K.G
    print(c(f"  Median return:      {summary['median_return_pct']:>+11.1f}%", g))
    print(f"  5th percentile:     {summary['p5_return_pct']:>+11.1f}%")
    print(f"  95th percentile:    {summary['p95_return_pct']:>+11.1f}%")
    print()
    print(c(f"  P(ruin):            {summary['prob_ruin']:>12.2%}", r_color))
    print(c(f"  P(2× capital):      {summary['prob_2x']:>12.2%}", K.G))
    print(c(f"  P(5× capital):      {summary['prob_5x']:>12.2%}", K.G))
    print(f"  P(losing money):    {summary['prob_loss']:>12.2%}")
    print()
    print(f"  Median max drawdown:{summary['median_max_drawdown_pct']:>11.1f}%")
    print(c(f"  Worst-case (P95):   {summary['p95_max_drawdown_pct']:>11.1f}%", K.Y))
    print()
    print(c("═" * 68, K.DIM))


# ============================================================
# 🚀  CLI
# ============================================================
def main():
    # Make sure terminal output handles emojis on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(
        description="Trading Strategy Monte Carlo Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--preset", type=str, help="Run a preset from presets.json")
    parser.add_argument("--list-presets", action="store_true",
                        help="List built-in strategy presets and exit")
    parser.add_argument("--win-rate", type=float, default=None,
                        help="Win probability, 0–1 (e.g. 0.45)")
    parser.add_argument("--risk", type=float, default=None,
                        help="Fraction of equity risked per trade (e.g. 0.01 = 1%%)")
    parser.add_argument("--rr", type=float, default=None,
                        help="Reward-to-risk ratio (e.g. 2.0 means win is 2× the risk)")
    parser.add_argument("--trades", type=int, default=200,
                        help="Number of trades per simulated path (default: 200)")
    parser.add_argument("--runs", type=int, default=10_000,
                        help="Number of Monte Carlo paths (default: 10,000)")
    parser.add_argument("--capital", type=float, default=10_000.0,
                        help="Starting capital (default: $10,000)")
    parser.add_argument("--name", type=str, default="Custom strategy",
                        help="Display name for the report")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip the HTML report (terminal output only)")
    args = parser.parse_args()

    presets = load_presets()

    if args.list_presets:
        if not presets:
            print(c("No presets found. Make sure presets.json is in the same folder.", K.R))
            sys.exit(0)
        print(c("📋  Available presets:", K.BOLD + K.CY))
        for key, s in presets.items():
            edge = s.edge_per_trade
            edge_color = K.G if edge > 0 else K.R
            print(f"  {c(key, K.B):<32}  {s.name}")
            print(f"    {c(s.description, K.DIM)}")
            print(f"    Win rate {s.win_rate:.0%}  ·  R:R {s.reward_to_risk}×  ·  "
                  f"Risk {s.risk_per_trade:.1%}  ·  "
                  f"Edge {c(f'{edge:+.2%}', edge_color)}")
            print()
        sys.exit(0)

    # Build the strategy
    if args.preset:
        if args.preset not in presets:
            print(c(f"⚠️  Unknown preset '{args.preset}'. Use --list-presets to see options.", K.R))
            sys.exit(1)
        strategy = presets[args.preset]
    else:
        if args.win_rate is None or args.risk is None or args.rr is None:
            print(c("⚠️  Provide --preset OR all of --win-rate / --risk / --rr.", K.R))
            print(c("    Run --list-presets to see built-in strategies.", K.DIM))
            sys.exit(1)
        strategy = Strategy(
            name=args.name,
            description="Custom strategy from CLI flags",
            win_rate=args.win_rate,
            risk_per_trade=args.risk,
            reward_to_risk=args.rr,
            n_trades=args.trades,
        )

    # Run simulation
    print(c(f"🎲  Running {args.runs:,} simulations of {strategy.n_trades} trades …",
            K.CY))
    result = simulate(strategy, n_runs=args.runs,
                      starting_capital=args.capital, seed=args.seed)
    summary = summarize(result)

    print_summary(strategy, summary, args.capital)

    # HTML report
    if not args.no_report:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_name = strategy.name.lower().replace(" ", "-").replace("/", "-")
        out = Path("reports") / f"{timestamp}-{safe_name}.html"
        path = render_report(result, summary, out)
        print(c(f"  📄 Report saved: {path.absolute()}", K.G))
        print()


if __name__ == "__main__":
    main()
