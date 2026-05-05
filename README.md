# 🎲 Trading Strategy Monte Carlo Simulator

Stress-test a trading strategy by running thousands of simulated equity curves &mdash; instead of looking at a single backtest and hoping it generalizes.

A single backtest is **one path through randomness.** Two strategies with the same expected value can have wildly different real-world outcomes due to luck and trade sequencing. Monte Carlo simulation gives you the **distribution** of outcomes, not just one sample &mdash; which is what professional risk managers actually use.

## Why I built it

I run a few MQL5 trading bots. The forward-test results look great on paper, but a single forward-test is a single coin flip &mdash; you can't tell if the bot is genuinely profitable or just got lucky. This tool answers questions like:

- *If I run this strategy 10,000 times, how often do I go broke?*
- *What's the realistic worst-case drawdown I should plan for?*
- *Is my "edge" actually big enough to survive a long losing streak?*

## What it does

Given a strategy described by:
- **Win rate** &mdash; probability of a winning trade (e.g. `0.45`)
- **Risk per trade** &mdash; fraction of equity put at risk on each trade (e.g. `0.01` = 1%)
- **Reward-to-risk ratio** &mdash; on a win, you make this many times your risk (e.g. `2.0`)
- **Number of trades** per simulated path (e.g. `200`)

…it runs *N* independent Monte Carlo paths (default 10,000) and produces:

| Output | What it tells you |
|---|---|
| **Median return** | The middle-of-the-road outcome you should expect |
| **5th / 95th percentile** | Realistic worst-case and best-case bands |
| **Probability of ruin** | Chance equity drops below 50% of starting capital |
| **P(2×) / P(5×)** | Chance of doubling, 5×-ing your account |
| **Median / 95th-percentile drawdown** | The drawdown you should plan for, and the one you should fear |

…plus a **self-contained HTML report** with three charts: the fan-chart of all simulated paths, the histogram of final returns, and the histogram of max drawdown.

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/trading-strategy-monte-carlo.git
cd trading-strategy-monte-carlo
pip install -r requirements.txt

# See built-in strategy presets
python monte_carlo.py --list-presets

# Run one
python monte_carlo.py --preset goshenfx-inspired

# Or run a custom strategy
python monte_carlo.py --win-rate 0.45 --risk 0.01 --rr 2.0 --trades 200 --runs 10000
```

The terminal prints a summary; an HTML report is saved to `reports/<timestamp>-<strategy>.html` and looks like this:

```
══════════════════════════════════════════════════════════════════
  🎲  Monte Carlo Result — GoshenFX-inspired (5-bar breakout)
══════════════════════════════════════════════════════════════════

  Runs:                     10,000
  Trades / run:                150
  Edge per trade:           +0.66%
  Starting capital:        $10,000

  Median return:           +131.4%
  5th percentile:           +12.3%
  95th percentile:         +428.7%

  P(ruin):                    0.21%
  P(2× capital):             62.40%
  P(5× capital):              7.85%
  P(losing money):            7.20%

  Median max drawdown:       18.5%
  Worst-case (P95):          37.8%
══════════════════════════════════════════════════════════════════
```

## Built-in presets

| Preset | What it models |
|---|---|
| `coinflip-1to1` | Pure coin flip with 1:1 R:R &mdash; no edge |
| `coinflip-2to1` | Coin flip with 2:1 R:R &mdash; edge from R:R alone |
| `scalp-aggressive` | High win rate (65%) with small wins relative to losses |
| `trend-follow` | Low win rate (40%) with big asymmetric wins |
| `goshenfx-inspired` | Modeled loosely on my own MQL5 breakout EAs |
| `overconfident-trader` | What happens when you risk too much per trade |
| `negative-edge` | Slight house edge against you (fees + slippage) |

Run `python monte_carlo.py --list-presets` to see full descriptions.

## CLI flags

| Flag | What it does |
|---|---|
| `--preset <name>` | Run a built-in preset |
| `--list-presets` | Show all built-in presets and exit |
| `--win-rate <0-1>` | Win probability for a custom strategy |
| `--risk <0-1>` | Fraction of equity risked per trade |
| `--rr <number>` | Reward-to-risk ratio |
| `--trades <int>` | Number of trades per simulated path (default: 200) |
| `--runs <int>` | Number of Monte Carlo paths (default: 10,000) |
| `--capital <$>` | Starting capital (default: $10,000) |
| `--seed <int>` | Random seed (for reproducible results) |
| `--no-report` | Skip the HTML report (terminal output only) |

## How it works (the math)

For each simulated path of `N` trades:

1. Generate `N` random uniform numbers; if `< win_rate`, the trade is a win.
2. On a win, multiply equity by `(1 + risk × reward_to_risk)`.
3. On a loss, multiply equity by `(1 - risk)`.
4. Track the running maximum and the drawdown from it.

The whole simulation is vectorized with NumPy &mdash; 10,000 runs × 200 trades = 2 million trade simulations finish in well under a second.

## Honest caveats

- **i.i.d. assumption.** Each trade is treated as independent of every other. Real markets have streaks (regime changes, momentum, news shocks) that violate this. Treat results as a *lower bound* on outcome variance.
- **Fixed parameters.** The simulator assumes win rate and R:R stay constant. In reality, market regimes change. Consider running multiple Monte Carlos &mdash; one per regime.
- **No fees / slippage built in.** Bake those into the win rate and R:R yourself, or extend the simulator (see "Extensions" below).
- **Compound returns only.** Equity is multiplied by `(1 ± risk)` per trade. No partial fills, no margin calls modeled separately, no portfolio-level constraints.

## Possible extensions

- Add fees / slippage as a fixed haircut per trade
- Sequence-correlation (use a Markov chain for win/loss instead of i.i.d.)
- Multiple strategies side-by-side comparison (overlay equity fans)
- Position-sizing strategies (Kelly, fixed-fractional, fixed-dollar)
- Export raw equity curves to CSV for downstream analysis

## Tech

- **Python 3.10+**
- **NumPy** for vectorized Monte Carlo (the core simulator is ~30 lines of array math)
- **matplotlib** for charts
- **No data science framework dependencies** &mdash; runs anywhere Python runs

## License

MIT &mdash; use it, fork it, ship something.

---

Built by **Carl Owen E. Belen** &middot; companion analysis tool for my [GoshenFX trading bots](https://github.com/YOUR-USERNAME) &middot; [Portfolio](https://github.com/YOUR-USERNAME)
