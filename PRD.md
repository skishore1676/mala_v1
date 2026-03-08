
---

## 1. Executive Summary
**Vision:** To build a state-of-the-art backtesting environment that models market price action as a physical object moving through a resistive medium, rather than a discrete series of random outcomes.
**Core Value Prop:** Unlike traditional backtesters that focus on P&L (Profit & Loss) of specific setups, the Kinematic Engine focuses on **Probability & Force**. It uses calculus-based derivatives (Velocity, Acceleration, Jerk) and Volume Profiling (Gravity) to determine the "Expected Move" and "Confidence Interval" of any given setup.

---

## 2. Problem Statement
Retail backtesting tools suffer from "Discrete State Fallacy." They test binary conditions (e.g., "RSI > 70") without context of the market's inertia.
* **Current Failure Mode:** A strategy buys a breakout, but fails to account for decelerating momentum (Negative Acceleration) into a resistance level (VPOC).
* **The Opportunity:** By applying kinematic physics equations to 1-minute aggregate data, we can filter out "false force" moves and identify high-probability "inertia" plays.

---

## 3. System Architecture

### 3.1 High-Level Stack
* **Language:** Python 3.10+
* **Data Source:** Polygon.io (Stocks & Indices) via API.
* **Storage Layer:** * Some kind of database that is efficient in storing and querring large volume locally. 
* **Compute Layer:** Pandas / NumPy (Vectorized Operations) or anything that improves effciency. 

---

## 4. Functional Requirements

### 4.1 Module A: The "Chronos" Data Pipeline
**Objective:** Efficiently ingest and store historical market data.
* **FR 1.1:** System must connect to Polygon.io REST API.
* **FR 1.2:** System must download 1-Minute Aggregates (OHLCV + VWAP) for configurable tickers (e.g., SPY, QQQ, IWM).
* **FR 1.3:** System must handle "Pagination" to download multi-year history (2+ years).
* **FR 1.4:** System must normalize data for Splits and Dividends (Adjusted Close).
* **FR 1.5:** Data must be saved locally to avoid redundant API calls. Checks: `if_exists(date) -> skip`.

### 4.2 Module B: The "Newton" Physics Engine
**Objective:** Pre-calculate kinematic properties of price action before any strategy logic is applied.
* **FR 2.1 - Velocity ($v$):** Calculate the 1st derivative of Price with respect to Time ($dP/dt$).
    * *Metric:* 1-period Rate of Change (ROC).
* **FR 2.2 - Acceleration ($a$):** Calculate the 2nd derivative ($dv/dt$).
    * *Metric:* Change in ROC. used to detect "exhaustion" (price rising, acceleration falling).
* **FR 2.3 - Jerk ($j$):** Calculate the 3rd derivative ($da/dt$).
    * *Use Case:* Detecting "Shock" events or trend reversals.
* **FR 2.4 - Gravity (VPOC):** Calculate the Volume Point of Control (VPOC) for the rolling lookback window (default: 4 hours).
    * *Logic:* VPOC acts as a magnet. If $Price \neq VPOC$, calculate "Distance to Mean."

### 4.3 Module C: The Strategy Agent (Configurable Logic)
**Objective:** A flexible logic gate that accepts "Physics States" and outputs "Probabilities."

#### User Story: The "EMA Momentum" Strategy
> "As a user, I want to test the 4/8/12 EMA stack to see if it predicts a 15-minute directional move."

* **Logic Gate 1 (Trend):** `EMA(4) > EMA(8) > EMA(12)`
    * *Physics Translation:* Velocity is Positive.
* **Logic Gate 2 (Location):** `Price > VPOC`
    * *Physics Translation:* Price has escaped "Gravity."
* **Logic Gate 3 (Force):** `Volume > Moving_Avg_Volume(20)`
    * *Physics Translation:* Mass is sufficient to sustain momentum.

### 4.4 Module D: The Output Oracle
**Objective:** Output a probabilistic surface, not just a trade log.
* **Metric 1:** **Maximum Favorable Excursion (MFE)**.
    * *Definition:* The highest price reached in the *next* 15 minutes after the signal.
* **Metric 2:** **Maximum Adverse Excursion (MAE)**.
    * *Definition:* The lowest price reached (Drawdown) in the *next* 15 minutes.
* **Metric 3:** **Confidence Score**.
    * *Formula:* `(Count of Wins / Total Signals)` where "Win" is defined as MFE > 2x MAE.

---

## 5. Technical Specifications (Data Structure)

The database schema (or Parquet column structure) will include:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | Datetime | UTC Time of the 1-min bar close. |
| `ticker` | String | e.g., "SPY" |
| `open`, `high`, `low`, `close` | Float | Price data. |
| `volume` | Int | Number of shares traded. |
| `velocity_1m` | Float | $P_t - P_{t-1}$ |
| `accel_1m` | Float | $V_t - V_{t-1}$ |
| `ema_4`, `ema_8`, `ema_12` | Float | Calculated Indicators. |
| `vpoc_4h` | Float | Most traded price in last 4 hours. |
| `forward_mfe_15` | Float | **Target Variable:** Max High in next 15 bars. |

---

## 6. Roadmap & Milestones

* **Phase 1 (Foundation):** Build `DataDownloader` class. Successfully pull 2 years of IWM 1-min data.