# Same-game correlation, in plain English

*A friendly companion to the technical write-up in `decisions_p7.md` §7.5. No
stats background needed.*

## The one-sentence version

When you bet on two things in the **same game**, they often rise and fall
together — so betting both isn't really two separate bets. This work measured
*which* pairs move together, by how much, and which only look like they do.

## Why this matters (a coin-flip story)

Imagine you like two bets in the same game: **the QB throws for a lot** and
**his top receiver catches for a lot.** Those aren't independent. If the QB has a
big day, the receiver almost certainly did too — the yards came from the same
throws. So putting both on your slip is closer to making **one big bet twice**
than making two different bets.

That matters for two reasons:

1. **Picking.** If your "top 5 bets of the week" are secretly five versions of
   *"this one game goes high-scoring,"* you don't have five edges — you have one
   edge you've quintupled your risk on. One bad game sinks all five.
2. **Bet sizing (later).** Parlays that combine legs from the same game
   ("same-game parlays") are only priced correctly if you know how the legs move
   together. Guess the correlation wrong and the parlay math is wrong.

## What we actually did (no jargon)

For every prop, we compared what the player **actually did** to what our model
**expected** — call that the "surprise." A +surprise means they beat the
projection; a −surprise means they fell short.

Then, game by game, we asked: **when player A over-performs, does player B tend
to over-perform too?** We did this across seven seasons of games and grouped the
answers by *type* of pair (QB-and-his-receiver, two-running-backs-on-opposite-
teams, a player's own two stats, and so on).

Three things kept us honest:

- **We only ever used the past to judge the present.** A correlation we'd "use"
  in 2024 was measured only from 2019–2023 — never peeking at the season it's
  meant to help with. (There's an automated test that fails if we ever cheat.)
- **We shrank shaky numbers toward zero.** If a pair type only showed up a
  handful of times, we don't trust its number and we pull it back toward "no
  relationship." Only pairs seen thousands of times, with a *consistent* story
  across seasons, keep their full strength.
- **We ignored "technically significant but tiny."** With this much data, even a
  meaningless 0.03 relationship looks "statistically real." We threw those out on
  purpose — a relationship has to be *big enough to matter*, not just detectable.

The strength is a number from **−1 to +1**: near **+1** = they move together
tightly, **0** = unrelated, near **−1** = when one goes up the other goes down.

## What we found

**Near-duplicates — a player's own stats (~0.76).** A receiver's receiving
*yards* and his *catches* are almost the same bet. Betting both is barely
different from betting one twice. Same for a QB's passing yards and pass
attempts, a running back's rushing yards and carries.

**Move together — a QB and his pass-catchers (~0.30).** When a quarterback has a
big passing day, his top receiver (~0.30) and tight end (~0.24) tend to come
along for the ride. Real, and steady every season. This is the main thing to
watch on a slip.

**Push against each other — run vs pass (about −0.08 to −0.10).** A team's
passing game and its own running game slightly *trade off* (more of one usually
means less of the other). And two running backs on **opposing** teams pull apart
(−0.10): the team that's winning runs the clock out while the team that's losing
throws. These pairs actually *balance* a slip rather than doubling up your risk.

**Shootouts — opposing quarterbacks (~+0.11).** In a back-and-forth game, both
QBs throw a lot. Mild but real.

**The myth that didn't hold up — two receivers on the same team (~0.03).** A lot
of daily-fantasy lore says to "stack" two wideouts from one team. In our data
they're basically **unrelated** — they're competing for the same passes, which
roughly cancels out the "this game is high-scoring" boost. We treat this as
**zero**. (Most two-touchdown-scorer pairings landed here too.)

## What this changes (and what it doesn't — yet)

**This job only measured and wrote things down.** It did **not** change any
picks or bet sizes. It hands the next two jobs a clean, tested reference they can
look up.

When those jobs use it, the practical upshot will be:

- **Don't stack near-duplicates.** A receiver's yards-over and catches-over
  shouldn't both count as separate edges — that's one bet wearing two hats.
- **Discount the QB + his receiver combo.** Two such "overs" are worth about 1.3
  bets, not 2. Size accordingly.
- **Leave the natural hedges alone.** A team's run-over next to its pass-under
  (or opposing running backs) actually *spreads* risk — those are fine together.
- **Only price same-game parlays for the handful of pairs we trust.** For
  everything else, we'll say "we don't know" rather than invent a number.

## Where the details live

- The full method, every number, and the honesty caveats: `decisions_p7.md`
  §7.5.
- The measured table you can regenerate any time:
  `reports/correlation_structure.md` (run `python3 scripts/fit_correlation.py`).
- The machine-readable reference the other jobs read:
  `data/correlation_structure.json`, via `nflvalue/correlation.py`.
