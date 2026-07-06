# Phase 7.5 — same-game prop correlation (walk-forward, shrunk)

Standardized residual correlation `(actual − proj mean)/proj sd`, pooled per pair type. Empirical-Bayes Fisher-z shrinkage toward 0 (τ²=0.07074). REAL = |shrunk ρ| ≥ 0.05 AND per-season sign stable — NOT t≥2 (n is huge; t flags economically-zero ρ as 'significant').

Synthetic-line caveat: residuals are vs the projection, not real prices.

## REAL correlation structure (consumable by 7.6/7.7)

| pair type | n pairs | ρ raw | ρ shrunk | per-season | sign-stable |
|---|---|---|---|---|---|
| `sameplayer|QB.pass~QB.pass` | 3,813 | +0.785 | **+0.783** | {2019: 0.7, 2020: 0.797, 2021: 0.757, 2022: 0.781, 2023: 0.791, 2024: 0.794, 2025: 0.814} | True |
| `sameplayer|TE.rec~TE.rec` | 4,202 | +0.775 | **+0.774** | {2019: 0.765, 2020: 0.791, 2021: 0.781, 2022: 0.769, 2023: 0.783, 2024: 0.784, 2025: 0.765} | True |
| `sameplayer|WR.rec~WR.rec` | 11,045 | +0.765 | **+0.764** | {2019: 0.785, 2020: 0.776, 2021: 0.754, 2022: 0.771, 2023: 0.744, 2024: 0.759, 2025: 0.783} | True |
| `sameplayer|RB.rush~RB.rush` | 5,974 | +0.764 | **+0.763** | {2019: 0.743, 2020: 0.738, 2021: 0.823, 2022: 0.777, 2023: 0.772, 2024: 0.753, 2025: 0.729} | True |
| `sameplayer|RB.rush~RB.td` | 11,948 | +0.353 | **+0.352** | {2019: 0.332, 2020: 0.366, 2021: 0.345, 2022: 0.339, 2023: 0.359, 2024: 0.352, 2025: 0.37} | True |
| `sameplayer|WR.rec~WR.td` | 22,090 | +0.332 | **+0.332** | {2019: 0.341, 2020: 0.337, 2021: 0.338, 2022: 0.304, 2023: 0.336, 2024: 0.364, 2025: 0.312} | True |
| `sameteam|QB.pass~WR.rec` | 11,660 | +0.297 | **+0.297** | {2019: 0.367, 2020: 0.288, 2021: 0.304, 2022: 0.286, 2023: 0.268, 2024: 0.301, 2025: 0.306} | True |
| `sameplayer|TE.rec~TE.td` | 8,404 | +0.269 | **+0.268** | {2019: 0.215, 2020: 0.26, 2021: 0.264, 2022: 0.316, 2023: 0.235, 2024: 0.245, 2025: 0.322} | True |
| `sameteam|QB.pass~TE.rec` | 4,443 | +0.245 | **+0.244** | {2019: 0.295, 2020: 0.266, 2021: 0.233, 2022: 0.215, 2023: 0.261, 2024: 0.2, 2025: 0.28} | True |
| `sameteam|QB.pass~WR.td` | 12,001 | +0.113 | **+0.113** | {2019: 0.134, 2020: 0.113, 2021: 0.111, 2022: 0.104, 2023: 0.132, 2024: 0.109, 2025: 0.098} | True |
| `opponent|QB.pass~QB.pass` | 2,014 | +0.110 | **+0.110** | {2020: 0.082, 2021: 0.16, 2022: 0.09, 2023: 0.087, 2024: 0.114, 2025: 0.105} | True |
| `opponent|RB.rush~RB.rush` | 4,919 | -0.100 | **-0.100** | {2019: -0.163, 2020: -0.031, 2021: -0.09, 2022: -0.114, 2023: -0.076, 2024: -0.083, 2025: -0.158} | True |
| `sameteam|QB.pass~TE.td` | 4,495 | +0.090 | **+0.090** | {2019: 0.209, 2020: 0.044, 2021: 0.062, 2022: 0.117, 2023: 0.122, 2024: 0.021, 2025: 0.116} | True |
| `sameteam|QB.pass~RB.rush` | 6,291 | -0.080 | **-0.080** | {2019: -0.073, 2020: -0.096, 2021: -0.122, 2022: -0.073, 2023: -0.083, 2024: -0.069, 2025: -0.05} | True |
| `opponent|QB.pass~WR.rec` | 11,568 | +0.052 | **+0.052** | {2019: 0.095, 2020: 0.039, 2021: 0.077, 2022: 0.031, 2023: 0.016, 2024: 0.052, 2025: 0.076} | True |
| `opponent|QB.pass~WR.td` | 11,912 | +0.051 | **+0.051** | {2019: 0.095, 2020: 0.086, 2021: 0.056, 2022: 0.055, 2023: 0.027, 2024: 0.013, 2025: 0.051} | True |

## NOISE (shrunk ~0 or sign-unstable — treated as 0 downstream)

| pair type | n pairs | ρ raw | ρ shrunk | why noise |
|---|---|---|---|---|
| `opponent|WR.rec~WR.td` | 34,528 | +0.033 | +0.033 | |ρ|<floor |
| `sameteam|WR.rec~WR.td` | 26,122 | +0.016 | +0.016 | |ρ|<floor |
| `opponent|RB.td~WR.td` | 25,836 | +0.016 | +0.016 | |ρ|<floor |
| `sameteam|RB.td~WR.td` | 25,764 | -0.038 | -0.038 | |ρ|<floor |
| `opponent|RB.td~WR.rec` | 25,093 | +0.006 | +0.006 | |ρ|<floor |
| `sameteam|RB.td~WR.rec` | 25,037 | +0.011 | +0.011 | |ρ|<floor |
| `opponent|RB.rush~WR.td` | 18,758 | -0.006 | -0.006 | |ρ|<floor |
| `sameteam|RB.rush~WR.td` | 18,657 | -0.005 | -0.005 | |ρ|<floor |
| `opponent|RB.rush~WR.rec` | 18,211 | -0.006 | -0.006 | |ρ|<floor |
| `sameteam|RB.rush~WR.rec` | 18,151 | -0.046 | -0.045 | |ρ|<floor |
| `opponent|WR.td~WR.td` | 17,771 | +0.026 | +0.026 | |ρ|<floor |
| `opponent|WR.rec~WR.rec` | 16,770 | +0.034 | +0.034 | |ρ|<floor |
| `opponent|RB.rush~RB.td` | 13,525 | -0.042 | -0.042 | |ρ|<floor |
| `sameteam|WR.td~WR.td` | 13,502 | -0.019 | -0.019 | |ρ|<floor |
| `opponent|TE.td~WR.td` | 13,273 | +0.029 | +0.029 | |ρ|<floor |
| `sameteam|TE.td~WR.td` | 13,166 | -0.035 | -0.035 | |ρ|<floor |
| `opponent|TE.rec~WR.td` | 13,128 | +0.022 | +0.022 | |ρ|<floor |
| `sameteam|TE.rec~WR.td` | 13,030 | +0.016 | +0.016 | |ρ|<floor |
| `opponent|TE.td~WR.rec` | 12,884 | +0.026 | +0.026 | |ρ|<floor |
| `sameteam|TE.td~WR.rec` | 12,818 | +0.016 | +0.016 | |ρ|<floor |
| `opponent|TE.rec~WR.rec` | 12,745 | +0.020 | +0.020 | |ρ|<floor |
| `sameteam|TE.rec~WR.rec` | 12,686 | +0.001 | +0.001 | |ρ|<floor |
| `sameteam|WR.rec~WR.rec` | 12,624 | +0.029 | +0.029 | |ρ|<floor |
| `sameteam|RB.td~TE.td` | 9,661 | -0.015 | -0.015 | |ρ|<floor |
| `opponent|RB.td~TE.td` | 9,618 | +0.015 | +0.015 | |ρ|<floor |
| `sameteam|RB.td~TE.rec` | 9,562 | +0.007 | +0.007 | |ρ|<floor |
| `opponent|RB.td~TE.rec` | 9,522 | +0.017 | +0.017 | |ρ|<floor |
| `opponent|RB.td~RB.td` | 9,354 | -0.008 | -0.008 | |ρ|<floor |
| `sameteam|QB.pass~RB.td` | 8,679 | +0.007 | +0.007 | |ρ|<floor |
| `opponent|QB.pass~RB.td` | 8,637 | +0.019 | +0.019 | |ρ|<floor |
| `sameteam|RB.rush~RB.td` | 8,355 | +0.001 | +0.001 | |ρ|<floor |
| `sameteam|RB.rush~TE.td` | 7,047 | +0.019 | +0.019 | |ρ|<floor |
| `sameteam|RB.rush~TE.rec` | 6,969 | -0.027 | -0.027 | |ρ|<floor |
| `opponent|RB.rush~TE.td` | 6,950 | -0.014 | -0.014 | |ρ|<floor |
| `opponent|RB.rush~TE.rec` | 6,882 | +0.001 | +0.001 | |ρ|<floor |
| `opponent|QB.pass~RB.rush` | 6,233 | +0.005 | +0.005 | |ρ|<floor |
| `sameteam|RB.td~RB.td` | 5,994 | -0.019 | -0.019 | |ρ|<floor |
| `opponent|TE.rec~TE.td` | 5,009 | +0.021 | +0.021 | |ρ|<floor |
| `opponent|QB.pass~TE.td` | 4,451 | +0.039 | +0.039 | |ρ|<floor |
| `opponent|QB.pass~TE.rec` | 4,408 | +0.023 | +0.022 | |ρ|<floor |
| `sameteam|RB.rush~RB.rush` | 2,708 | +0.051 | +0.051 | sign unstable |
| `opponent|TE.td~TE.td` | 2,527 | +0.028 | +0.028 | |ρ|<floor |
| `opponent|TE.rec~TE.rec` | 2,482 | -0.013 | -0.013 | |ρ|<floor |
| `sameteam|TE.rec~TE.td` | 2,192 | +0.003 | +0.003 | |ρ|<floor |
| `sameteam|TE.td~TE.td` | 1,116 | -0.004 | -0.004 | |ρ|<floor |
| `sameteam|TE.rec~TE.rec` | 1,076 | -0.005 | -0.005 | |ρ|<floor |
| `sameteam|QB.pass~QB.pass` | 393 | -0.423 | -0.410 | sign unstable |

Artifact: `/sessions/elegant-brave-heisenberg/mnt/nflgambling/fablesfable/data/correlation_structure.json` (production shrunk ρ + walk-forward slices).

