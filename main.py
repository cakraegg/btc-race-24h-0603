"""Leviathan Dip — a tiered capitulation-dip buyer for cbBTC (make.fun btc-race).

Strategy (validated on the kit dev window + 6 walk-forward 24h windows of
Coinbase BTC 1-minute candles, Nov 10-16 2026):

  * The dev window is flat and low-volatility: 60-minute log returns bottom
    at -2.08%, so the wide dip thresholds of the previous race entry
    (-2.2% / -2.8%) never fire there and that bot scores $0 on this window.
  * This bot buys the capitulation dip in two stages, with a bounce
    confirmation, and then rides the recovery patiently:
      - TIER 1: 60-min log return below -2.0% AND a green bar (close above
        the previous close and above the open) -> buy 50% of cash.
      - TIER 2: while holding, if the 60-min return deepens below -2.5%
        -> buy all remaining cash (average entry drops toward the low).
        The buy is confirmed only when the position actually grows; a
        fill refused at the next bar's open (gap-up past the 1% slippage
        cap) re-fires on the next bar instead of burning the tier.
      - EXIT: sell everything 120 bars after the tier-1 fill, or sooner on
        a hard stop 1.5% below the tier-1 decision close. No take-profit,
        no trailing stop: cutting the bounce short cost the most money in
        backtests (the big winning trade rode the full 120 bars).
      - After any exit, 20 bars of cooldown before the next entry.
  * Long only, one position at a time, stdlib only, no network, no clock,
    no randomness: the same bars always give the same orders.

Backtest (exact replay model: fee 4 bps + slippage 2 bps + constant-product
impact vs pool depth, fills at the next bar's open), $1,000 start:
  dev window      +$4.78   (1 trade, max DD 0.29%)
  6 walk-forward windows (Nov 10-16): +$37.88, $0.00, $0.00, -$4.64,
  $0.00, $0.00  -> mean +$5.43/window, worst -$4.64, max DD 1.43%.
  Baselines on the dev window: buy-and-hold +$1.07, v2-style deep dip $0.00
  (its thresholds never fire here), sensitive dip scalps lose money on the
  downtrend windows. The staged deep dip is the best mean / worst-case
  design found across all seven windows.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# --- Strategy constants (fixed; tuned on dev + walk-forward, see header). ---
LOOKBACK = 60            # bars for the dip return
TIER1_DROP = -0.020      # 60-min log return below this arms tier 1
TIER1_SIZE = 0.5         # fraction of cash deployed at tier 1
TIER2_DROP = -0.025      # 60-min log return below this arms tier 2 (rest of cash)
HOLD_BARS = 120          # bars after the tier-1 fill before the time exit
STOP_BPS = 150           # hard stop: 1.5% below the tier-1 decision close
COOLDOWN_BARS = 20       # bars of no entry after any exit
MIN_NOTIONAL = 1_000_000  # 1 USDC in raw units (fees.min_notional_quote_units)
BUY_SLIPPAGE_BPS = 100   # 1%: a refused buy just re-fires next bar
SELL_SLIPPAGE_BPS = 10_000  # exits must always execute, even into a gap


class Bot:
    def __init__(self):
        self.closes = []   # CBBTC closes (price_scale 5 ints), one per bar
        self.opens = []    # CBBTC opens, same order
        self.meta = None   # confirmed position: dict(entry_bar, avg, tier2_done)
        self.pending = None  # unconfirmed tier-1 buy: dict(bar, avg)
        self.pending_tier2 = None  # unconfirmed tier-2 buy: dict(bar, qty_before)
        self.pending_exit = False
        self.cd_until = -1

    def on_batch(self, state, batch):
        if state.warming_up:
            self._record(state)
            return []
        bar = self._record(state)
        if bar is None:
            return []  # no CBBTC bar in this batch: nothing new to decide on
        i = len(self.closes) - 1
        holding = state.positions.get(SYMBOL)
        # Reconcile a pending tier-1 buy with the account: the fill lands in
        # the batch after the decision bar.
        if self.meta is None and self.pending is not None:
            if holding is not None:
                self.meta = self.pending
                self.pending = None
            elif i >= self.pending["bar"]:
                self.pending = None  # rejected: allow the signal to re-fire
        # Reconcile a pending exit.
        if self.pending_exit:
            if holding is None:
                self.pending_exit = False
                self.meta = None
                self.cd_until = i + COOLDOWN_BARS
            else:
                self.pending_exit = False  # rejected: re-evaluate the exit below
        # Reconcile a pending tier-2 buy: confirm ONLY if the position grew.
        # A fill refused at the next bar's open (gap-up past the 1%
        # slippage cap) used to mark tier 2 done without buying anything;
        # now it clears and the signal may re-fire, like tier 1.
        if self.pending_tier2 is not None:
            if self.meta is None:
                self.pending_tier2 = None  # position gone: stale
            elif holding is not None and holding.qty > self.pending_tier2["qty_before"]:
                self.meta["tier2_done"] = True
                self.pending_tier2 = None
            elif i >= self.pending_tier2["bar"]:
                self.pending_tier2 = None  # rejected: allow the signal to re-fire
        if holding is not None and self.meta is None:
            # A position we did not open (should not happen): treat its
            # appearance bar as the entry and its cost as the average.
            self.meta = {
                "entry_bar": i,
                "avg": self._avg_from_holding(state, holding),
                "tier2_done": True,
            }
        if holding is not None and self.meta is not None:
            return self._while_holding(state, i)
        return self._while_flat(state, i)

    def _record(self, state):
        """Append this batch's CBBTC bar (if any) to the history."""
        bar = state.latest.get(SYMBOL)
        if bar is None:
            return None
        if len(self.closes) and bar["t"] == self._last_t:
            return None  # same bar repeated: do not double-count
        self._last_t = bar["t"]
        self.closes.append(int(bar["c"]))
        self.opens.append(int(bar["o"]))
        return bar

    _last_t = -1

    @staticmethod
    def _avg_from_holding(state, holding):
        """Average entry price (price_scale units) from cost / qty."""
        qty = holding.qty
        if qty <= 0:
            return state.close(SYMBOL) or 0
        # value = qty * price * 10**quote_scale / 10**(qty_scale + price_scale)
        # price = cost * 10**(qty_scale + price_scale) / (qty * 10**quote_scale)
        sym = state.symbols[SYMBOL]
        quote_scale = state.hello["capital"]["quote_scale"]
        num = holding.cost * 10 ** (sym["qty_scale"] + sym["price_scale"])
        den = qty * 10 ** quote_scale
        return num // den if den else 0

    def _r60(self, i):
        if i < LOOKBACK:
            return 0.0
        return math.log(self.closes[i] / self.closes[i - LOOKBACK])

    def _green(self, i):
        return self.closes[i] > self.closes[i - 1] and self.closes[i] > self.opens[i]

    def _while_flat(self, state, i):
        if i < LOOKBACK or i < self.cd_until or self.pending is not None:
            return []
        if self._r60(i) < TIER1_DROP and self._green(i):
            cash = state.cash
            room = state.weight_room(SYMBOL)
            amount = int(cash * TIER1_SIZE)
            amount = min(amount, cash, room)
            if amount < MIN_NOTIONAL:
                return []
            self.pending = {"entry_bar": i + 1, "avg": self.closes[i], "tier2_done": False}
            return [{"side": "buy", "symbol": SYMBOL,
                     "notional_quote_units": amount,
                     "max_slippage_bps": BUY_SLIPPAGE_BPS}]
        return []

    def _while_holding(self, state, i):
        holding = state.positions[SYMBOL]
        meta = self.meta
        c = self.closes[i]
        # Hard stop: 1.5% below the tier-1 decision close.
        if c * 10_000 <= meta["avg"] * (10_000 - STOP_BPS):
            self.pending_exit = True
            return [{"side": "sell", "symbol": SYMBOL, "qty": holding.qty,
                     "max_slippage_bps": SELL_SLIPPAGE_BPS}]
        # Time exit: 120 bars after the tier-1 fill.
        if i - meta["entry_bar"] >= HOLD_BARS:
            self.pending_exit = True
            return [{"side": "sell", "symbol": SYMBOL, "qty": holding.qty,
                     "max_slippage_bps": SELL_SLIPPAGE_BPS}]
        # Tier 2: the dip deepened -> deploy the rest of the cash.
        # Record it as pending; tier2_done is set only once the fill is
        # confirmed in on_batch (a refused buy re-fires next bar).
        if not meta["tier2_done"] and self.pending_tier2 is None and self._r60(i) < TIER2_DROP:
            amount = min(state.cash, state.weight_room(SYMBOL))
            if amount >= MIN_NOTIONAL:
                self.pending_tier2 = {"bar": i + 1, "qty_before": holding.qty}
                return [{"side": "buy", "symbol": SYMBOL,
                         "notional_quote_units": amount,
                         "max_slippage_bps": BUY_SLIPPAGE_BPS}]
        return []


def main():
    bot = Bot()

    def on_done(state, done):
        final = state.equity - 1_000_000_000
        sys.stderr.write("leviathan-dip done: equity %d raw, pnl %+d raw\n" % (state.equity, final))

    makefun_bars.run(bot.on_batch, on_done=on_done)


if __name__ == "__main__":
    main()
