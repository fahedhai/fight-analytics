# Muay Thai / Combat Sports Fight Analytics

**Author:** Fahed Haidar — Professional Muay Thai fighter & IT/Data Science student

## Overview
As a professional Muay Thai fighter, I wanted to see whether the striking
and physical data behind fighters actually backs up conventional fight
wisdom — does reach really matter, does stance affect striking accuracy,
how does age relate to performance? This project explores those questions
using UFC fighter and fight data as a proxy for striking-based combat sports.

## Data
Source: [UFC-Fight Historical Data From 1993 To 2021](https://www.kaggle.com/datasets/rajeevw/ufcdata) (Kaggle, rajeevw)

Files used:
- `raw_fighter_details.csv` — career-level fighter profiles (height, reach, stance, SLpM, striking/grappling accuracy)
- `data.csv` — fight-by-fight records with Red/Blue corner stats and outcomes (~6,000 fights)

`clean_data.py` reshapes `data.csv` from one-row-per-fight (Red/Blue columns)
into one-row-per-fighter-per-fight (`fights_long.csv`, ~12,000 rows), which
makes fighter-level questions (win rate by stance, by age, etc.) much easier
to answer.

## Project Structure
```
muaythai-fight-analytics/
├── data/               # raw and cleaned CSVs (not committed if large)
├── notebooks/          # exploratory Jupyter notebooks + generated figures
├── src/
│   ├── clean_data.py   # cleans raw CSVs into tidy format
│   └── analyze.py      # generates exploratory charts
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python src/clean_data.py     # produces data/fights_long.csv, fighter_profiles.csv
python src/analyze.py        # produces 5 charts in notebooks/figures/
```

## Questions Explored
1. Does stance (orthodox vs southpaw vs switch) affect win rate and striking accuracy?
2. Is there a relationship between reach and striking output?
3. How is fighter age distributed, and does win rate decline with age?
4. Where do strikes actually land — head, body, or leg?

## Key Findings
- **Southpaws and switch fighters win slightly more often than orthodox fighters**
  (~51.7% vs ~48.6% win rate). This lines up with the common "southpaw advantage"
  theory in striking sports — orthodox fighters see fewer southpaw opponents in
  training, so southpaws may have a stylistic edge. Sample size caveat: southpaws
  are a minority of fighters (2,396 vs 9,068 orthodox rows), so this needs a
  significance check before reading too much into it.
- **Reach shows a positive but noisy relationship with strikes landed** — longer
  reach fighters land more significant strikes on average, but there's wide
  variance, meaning reach alone is a weak predictor. Technique and output clearly
  matter more than physical measurements alone.
- **Younger fighters win more.** Win rate drops fairly steadily from ~55% (18-24)
  to ~38% (37+). This is expected (athleticism, recovery, reaction time), but
  it's a useful sanity check that the data behaves the way real fight physiology
  predicts.
- **Striking accuracy by stance is nearly identical** across orthodox and
  southpaw (~45% median), with switch fighters slightly higher (~48%) — stance
  seems to matter more for win rate than raw accuracy.
- **64% of significant strikes target the head**, 20% body, 16% legs. As a Muay
  Thai fighter, this stands out — Muay Thai scoring rewards a much heavier mix of
  leg kicks and body strikes, so this head-heavy distribution reflects MMA's
  scoring/finishing incentives (head strikes are more likely to end a fight)
  rather than technical striking priorities in a pure striking sport like Muay Thai.

## What I'd Improve Next
- Extend to fight-outcome prediction (Project 2)
- Bring in Muay Thai-specific domain knowledge to interpret striking metrics
- Explore pose-estimation on fight footage (Project 3)

## About Me
I bring a fighter's perspective to this analysis — [one line about your
own fighting background / why this data matters to you].
