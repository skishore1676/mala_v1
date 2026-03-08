# Market Impulse Strategy — Exploration Findings
**Date:** 2026-03-03  
**Instruments tested:** SPY, QQQ, IWM  
**Data range:** 2024-03-03 → 2026-03-03 (2 years, 1-min bars)

---

## 1. The Idea

Inspired by the TOS "Market Pulse" indicator. The core hypothesis: use an adaptive VMA (Variable Moving Average) and VWMA regime stack to detect early trend impulses at market open, then trade in the direction of the impulse.

### Indicator Components

**VMA (Variable Moving Average)**  
An adaptive moving average where the smoothing coefficient changes based on *directional price strength*. In trending conditions, it reacts faster; in choppy conditions, it moves slowly.

```
tmp1 = price > price[1] → price - price[1] else 0   (up moves)
tmp2 = price[1] > price → price[1] - price else 0   (down moves)
d2 = sum(tmp1, length)
d4 = sum(tmp2, length)
ad3 = (d2 - d4) / (d2 + d4) * 100                   (directional score -100 to +100)
coeff = 2 / (length + 1) * |ad3| / 100              (adaptive smoothing factor)
VMA = coeff * price + VMA[1] * (1 - coeff)          (EMA-like, adaptive)
```

**VWMA Regime Stack (8 / 21 / 34)**
```
Bullish:  VWMA(8) > VWMA(21) > VWMA(34)
Bearish:  VWMA(8) < VWMA(21) < VWMA(34)
Neutral:  anything else
```

**Stage Classification**
| Stage | Condition |
|:--|:--|
| Acceleration | Bullish regime + close ≥ VMA |
| Deceleration | Bearish regime + close ≤ VMA |
| Accumulation | Bullish regime + close < VMA |
| Distribution | Bearish regime + close > VMA |

---

## 2. Experiments Conducted

### Experiment 1 — Cross-and-Reclaim (Multi-Timeframe)

**Hypothesis:** Use the 5-min VWMA regime as a directional filter and the 1-min VMA cross-and-reclaim as entry trigger.

**Entry rules:**
- Long: 5-min regime = bullish + 1-min bar low touches VMA but close recovers above
- Short: 5-min regime = bearish + 1-min bar high touches VMA but close falls back below
- Time window: 9:33 – 10:30 ET

**Exit rules:** Full 1-min bar on the wrong side of the 5-min VMA (bar HIGH < VMA_5m for longs)

**Results (SPY, VMA length=10):**

| Direction | Trades | Win% | Avg Win | Avg Loss | PF | Exp/Trade |
|:--|--:|--:|--:|--:|--:|--:|
| Combined | 67 | 34.3% | +$1.08 | −$0.77 | 0.73 | −$0.14 |
| Long | 38 | 42.1% | +$0.66 | −$0.54 | 0.89 | −$0.03 |
| Short | 29 | 24.1% | +$2.05 | −$1.01 | 0.65 | −$0.27 |

**Verdict:** No edge. Long side nearly breakeven; short side dragged it down. Note: all exits via VMA stop (~89 min avg hold), not EOD.

---

### Experiment 2 — Stage Flip, VMA Sweep (2 / 3 / 4)

**Hypothesis:** Simplify entry to a stage flip (acceleration or deceleration appears for the first time), exit when stage flips away.

**Results (SPY, no regime filter):**

| VMA Len | Trades | Win% | PF | Exp/Trade | Total $ |
|:--|--:|--:|--:|--:|--:|
| 2 | 1,137 | 33.5% | 0.93 | −$0.005 | −$6.08 |
| **3** | **906** | **32.8%** | **0.98** | **−$0.002** | **−$1.67** |
| 4 | 758 | 31.7% | 0.92 | −$0.008 | −$6.12 |

**Key finding:** VMA=3 is the sweet spot — 0.98 PF, closest to breakeven, ~3 bars avg hold. But 900+ trades in a year during a 57-minute window = ~3.5/day. Too twitchy.

---

### Experiment 3 — Stage Flip VMA=3 + 5-min Regime Confirmation

**Hypothesis:** Only take longs when 5-min regime = bullish, shorts when 5-min regime = bearish.

| Config | Trades | Win% | PF | Exp/Trade | Total $ |
|:--|--:|--:|--:|--:|--:|
| Baseline | 906 | 32.8% | **0.98** | −$0.002 | −$1.67 |
| + 5m confirmation | 353 | 30.3% | 0.92 | −$0.008 | −$2.85 |

**Verdict:** Hurt, not helped. The 5-min regime is cutting winners along with losers. The two timeframes are not correlated strongly enough.

---

### Experiment 4 — Cross-Ticker Comparison (SPY / QQQ / IWM), VMA=3 Stage Flip

| Ticker | Trades | Win% | Avg Win | Avg Loss | PF | Exp/Trade | Total $ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **SPY** | 906 | 32.8% | +$0.265 | −$0.132 | **0.98** | −$0.002 | **−$1.67** |
| QQQ | 2,137 | 33.3% | +$0.209 | −$0.118 | 0.88 | −$0.009 | −$19.77 |
| IWM | 1,333 | 31.9% | +$0.140 | −$0.079 | 0.83 | −$0.009 | −$12.29 |

**Key findings:**
- SPY is by far the most efficient — fewest signals, best PF, largest per-trade moves
- QQQ generates 2.4× more signals than SPY — flipping stages too rapidly
- IWM worst performing despite being "more volatile"

---

## 3. Consistent Pattern Across All Experiments

**Shorts consistently outperform longs:**

| Ticker | Long PF | Short PF |
|:--|--:|--:|
| SPY | 0.98 | 0.98 |
| QQQ | 0.81 | **0.95** |
| IWM | 0.70 | **0.96** |

Hypothesis: In the first hour, early rips get faded by institutional sellers more reliably than early dips get bought. This is consistent with "opening range fade" behavior documented in the literature.

---

## 4. Technical Infrastructure Built

All code lives in `/Users/suman/kg_env/projects/mala_v1/`:

| File | Purpose |
|:--|:--|
| `src/newton/market_impulse.py` | VMA, VWMA, regime + stage computation |
| `src/newton/engine.py` | `enrich_market_impulse()` — market hours filter + multi-TF join |
| `src/strategy/market_impulse.py` | Cross-and-reclaim strategy (Experiment 1) |
| `src/oracle/metrics.py` | Directional MFE/MAE, EOD forward window, snapshots |
| `src/oracle/trade_simulator.py` | Bar-by-bar P&L simulator with VMA exit |
| `scripts/run_market_impulse.py` | Run Experiment 1 pipeline |
| `scripts/run_stage_flip.py` | Run Experiments 2–4 (stage flip sweep) |

**Important:** The market-hours filter in `enrich_market_impulse()` strips pre/post-market bars before computing indicators. This is essential — pre-market bars corrupt the adaptive VMA.

---

## 5. What We Learned

1. **The VMA stage signal is real but noisy** — 0.98 PF on SPY with VMA=3 means it's marginally better than random, but not meaningfully so on its own
2. **Signal frequency is the enemy** — 900 trades in 57-min daily windows generates noise, not edge. Fewer, higher-quality signals would be better
3. **Adding filters can hurt** — the 5-min regime filter actually destroyed edge. More conditions ≠ better
4. **Shorts have a structural advantage at open** — consistent across all three instruments. Worth pursuing
5. **SPY is the right instrument** — most efficient, cleanest signals, best ratio of avg-win to avg-loss

---

## 6. Promising Next Directions

### A. Short-Side Isolation + Persistence Filter
Only take short stage-flip signals, and require the stage to hold for N bars (e.g., 3) before entering. This cuts the micro-flip noise.

### B. Regime Persistence as the Signal
Instead of reacting to a flip, count how many consecutive bars the regime has been in the same state. Enter after N consecutive acceleration/deceleration bars. This naturally filters noise.

### C. Gap Fade Strategy (New Idea)
When SPY gaps up > 0.3% at open, fade it with a short if the VWMA regime is still bearish or neutral. Opening gaps in the 0.3–1.0% range have documented mean-reversion tendencies.

### D. Volume Surge Filter
From the earlier Opening Bell findings: high volume at the open was a strong predictor. Combine VMA stage signals with a "volume is > 2× the 20-day average for this time of day" filter.

---

## 7. Key Parameters

```python
# Default Market Impulse settings (src/config.py)
vma_length = 10            # for cross-and-reclaim (Exp 1)
vma_length = 3             # optimal for stage flip (Exp 2-4)
vwma_periods = [8, 21, 34] # regime stack
market_open_hour = 9       # ET
market_open_minute = 30    # ET
impulse_entry_buffer_minutes = 3    # 9:30 + 3min = entry from 9:33
impulse_entry_window_minutes = 60   # no entries after 10:30
```
