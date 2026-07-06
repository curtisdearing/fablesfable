# Bet sizing, in plain English

*A friendly companion to `decisions_p7.md` §7.7. No math background needed.*

## The one-sentence version

Even a good model goes broke if it bets too big, so this figures out a **safe,
recommended amount** to put on each pick — small, cautious, and adjusted for the
fact that some picks overlap. **It only suggests. It never places a bet.**

## Why sizing is the dangerous part

Picking winners and *sizing* bets are two different skills. You can be right more
often than the market and still lose everything by betting too much on a cold
streak. The famous "Kelly" formula says exactly how much to bet to grow money
fastest — but it assumes you know your true win rate *perfectly*. We don't. Our
win rate is an educated guess, and our guess could be too rosy.

So the whole design here is **deliberate caution**: bet as if our edge is smaller
and shakier than it looks, because it probably is.

## How it decides an amount (in plain steps)

1. **Is there an edge at all?** We compare our model's chance of the pick hitting
   to the price the book is offering. If we're not actually getting a better deal
   than fair, **we bet nothing.** No edge, no bet.
2. **Assume the edge is smaller than it looks.** The betting market is very good
   at pricing games. So we cut our estimated edge roughly in half before doing
   anything — a humility tax.
3. **Bet a small fraction of that.** We then bet only a *quarter* of what the
   textbook "grow fastest" formula would say. Quarter-Kelly is the standard
   cautious setting; it trades a little growth for a lot less risk.
4. **Don't double-count overlapping picks.** If two picks tend to win or lose
   together — say a quarterback's passing over and his receiver's catching over —
   they're really one bet wearing two hats. We shrink each so the pair doesn't
   secretly become a giant single wager. (This uses the correlation work from the
   previous job.) Picks that *hedge* each other are left alone — they're helpful.
5. **Hard ceilings.** No single pick gets more than **2% of the bankroll**, and
   all picks on a given day together are capped at **10%**. So one bad night can't
   sink you.

We measure size in "units," where **1 unit = 1% of your bankroll.** A typical
pick comes out around **half a unit to one unit** — small on purpose.

## Does the caution actually help? (we tested it)

We simulated thousands of seasons at *realistic* win rates (52–58%, the honest
range against real sharp lines — **not** the flashy 66–68% we hit against our own
practice lines, which would be fantasy).

The picture, starting from 100 units:

- **Our cautious rule** grows the bankroll steadily and, even in a bad-luck
  season (worst 1-in-20), the deepest dip is only about **10–15%**. Going broke
  basically never happened.
- **Betting four times bigger** (plain quarter-Kelly with no humility tax) grows
  faster when you're lucky but can swing down **30–48%** — stomach-churning, and
  a real risk of a long dark stretch.
- **At a true 52.4% (break-even), our rule simply doesn't bet.** No clever sizing
  can invent an edge that isn't there — and it doesn't pretend to.

The trade is intentional: we give up some upside to almost never blow up. For an
edge we only *estimate*, that's the right trade.

## The honest limits

- **These are suggestions, full stop.** The tool hands you a recommended size and
  a risk readout. You decide. It never touches money.
- **The numbers assume the edge is real.** Whether it *is* real is a separate
  question, answered by the closing-line-value experiment (the earlier CLV work),
  not by this sizing math. Sizing tells you *how much to risk if* the edge holds
  up — it can't tell you *that* it does.
- **Practice-line results don't transfer 1-to-1.** Everything is still graded
  against our synthetic reference lines until real prices accrue, so treat the
  growth figures as "what a plausible real edge would look like," not a promise.

## Where the details live

- Full rule, settings, and the simulation table: `decisions_p7.md` §7.7.
- The bankroll simulation you can re-run: `reports/staking_mc.md`
  (`python3 scripts/staking_mc.py`).
- The sizing code (with the "advisory only" banner right at the top):
  `nflvalue/staking.py`.
