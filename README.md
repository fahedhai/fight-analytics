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
- ~~Extend to fight-outcome prediction (Project 2)~~ Done, see below
- Bring in Muay Thai-specific domain knowledge to interpret striking metrics
- Explore pose-estimation on fight footage (Project 3)

## About Me
I bring a fighter's perspective to this analysis — [one line about your
own fighting background / why this data matters to you].

---

# Project 2: Fight Outcome Prediction

## Overview
Building on Project 1, this extends the analysis into a predictive model:
given two fighters' pre-fight career stats, can we predict who wins?

## Approach
- **Framing:** for each fight, compute Red-minus-Blue differences across
  striking, grappling, and physical stats (reach, age, win streaks, etc.)
  going into the fight. This is a standard head-to-head sports modeling
  setup — the model learns from *relative* advantages, not absolute stats.
- **Target:** `red_win` (1 if Red corner won). Draws/no-contests dropped.
- **Important baseline:** Red corner wins **65%** of fights in this dataset
  (a known quirk — the higher-ranked/favored fighter is usually assigned
  Red). So the real bar for a useful model isn't 50%, it's 65%.
- **Models compared:** Logistic Regression, Random Forest, Gradient Boosting.

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Baseline (always predict Red) | 0.65 | 0.50 |
| Logistic Regression | 0.664 | 0.638 |
| Random Forest | 0.663 | 0.657 |
| **Gradient Boosting** | **0.678** | **0.675** |

## Key Findings
- **Age difference is the single strongest predictor** — this lines up
  directly with Project 1's finding that win rate declines steadily with
  age. Good cross-check that both analyses agree.
- Striking output/accuracy differences, takedowns landed, and cage control
  time are the next-most-important features — technical output matters more
  than raw physical measurements like reach.
- **Honest limitation:** the confusion matrix shows the model is still
  heavily biased toward predicting "Red wins" — it correctly identifies
  542/553 Red wins but only 21/296 Blue (underdog) wins. This means the
  model is mostly leaning on the built-in Red-corner base rate rather than
  confidently picking underdog winners. This is a class imbalance problem,
  and gradient boosting only modestly outperforms just guessing "Red" —
  a realistic result, not a triumphant one.

## What I'd Try Next
- Handle class imbalance directly (`class_weight="balanced"`, SMOTE, or
  training on a rebalanced sample) so the model has to actually learn
  underdog patterns rather than defaulting to the majority class.
- Try removing corner assignment entirely and instead predict "Fighter A
  vs Fighter B" symmetrically, to strip out any Red/Blue bias altogether.
- Add rolling recent-form features (last 3 fights) rather than career
  averages, since a fighter's current form likely matters more than
  lifetime stats.

## How to Run
```bash
python src/build_features.py   # produces data/model_features.csv
python src/train_model.py      # produces model comparison + feature importance charts
```


Visual Studio Code 1.139

Show release notes after an update

Follow us on LinkedIn, X, Bluesky, Instagram | View online

Release date: September 23, 2026

Update 1.139.1: The update addresses these issues.

Already installed? Use Check for Updates in VS Code. For upcoming features, use the Insiders build.

Release highlights
This release makes large agent session lists faster, extends Dev Container support to remote projects, and improves everyday editing.

Remote Dev Container sessions: Run agents inside your project's Dev Container on SSH, Tunnel, and WSL hosts.

Session list improvements: Load large session lists faster, fit more sessions on screen, and in-place session renaming.

Editor experience: Identify wrapped lines at a glance and avoid duplicate closing brackets as you type.

Happy Coding!

In this update
Release highlights
Agents
Chat
Editor experience
Proposed APIs
Deprecated features and settings
Notable fixes
Thank you
Agents
The agent host runs agent harnesses in a dedicated process based on the Agent Host Protocol (AHP), so you can connect to the same session from multiple VS Code windows. Learn more about its architecture and workflows in the agent host blog post.

Run agent sessions in Dev Containers on remote hosts
Setting: chat.agentHost.devContainer.enabled (Agents Window only)

Let agents build and test your remote project with the right tools and dependencies, without duplicating toolchain setup on your laptop or the remote host. This release extends Dev Container sessions from local folders to projects on SSH, Tunnel, and WSL hosts.

To get started, enable chat.agentHost.devContainer.enabled and select Use Dev Container from the folder menu in the Agents Window. The remote folder must have a supported Dev Container configuration, and Docker must be available on the remote host.

Note: Dev Container sessions are rolling out gradually, so the setting might not be enabled by default for you yet. You can enable the setting manually to try the feature now.

Faster session list loading
VS Code loads and refreshes large agent session lists faster. The agent host keeps lightweight session and chat metadata in a central catalog instead of opening every conversation database each time the list is built. Full conversation content remains isolated in the individual session and chat databases.

The improvement grows with the number of sessions because the previous approach did work in proportion to your session count. Measured with around 645 sessions on a development machine:

Operation	Before	After	Improvement
First session listing after launch	1.3 seconds	0.1 seconds	About 12x faster
Refresh the session list	0.6 seconds	0.15 seconds	About 4x faster
If you have few sessions, expect a smaller difference. Sessions created before this release are migrated automatically in the background.

Compact sessions list
Fit more sessions in the sessions list by enabling Compact View in the sessions list view of the Agents Window.

Compact rows show the session title at rest and reveal workspace details when you hover over or focus the row. A row expands when the session needs input or approval, so these requests remain visible.

Progress also appears on the row for the chat that owns the work. When you collapse a session, the parent row summarizes progress from its hidden chats.


Filter empty session groups
Disable Empty Groups from Filter Sessions to hide empty custom groups and the empty Chats section. This preference is stored in your profile and resets with the other sessions list filters.

Rename sessions and chats in place
Rename a session or nested chat directly in the sessions list. Double-click its title, use the Rename context menu action, or focus the row and press sessions.sessionHeader.rename for a session or sessions.chatCompositeBar.renameChat for a nested chat. Inline validation prevents blank titles, and canceling restores the previous title.

Choose how chats appear in a session (Preview)
Setting: sessions.showChatTabs (Agents Window only)

An agent session can contain multiple chats, each representing a different conversation or context. When a session contains multiple chats, choose the presentation that best fits your workflow from the session header menu:

Multiple shows each chat on its own tab.
Single shows only the active chat and hides the tab bar.
Switching presentations preserves your open chats, active chat, and conversation state. In Single mode, chats that you explicitly open to the side remain independent panes with their own header actions.


Chat
Pet naming contest update (Experimental)
Thank you to everyone who submitted a name for the VS Code pet. The naming contest closed on September 17, 2026, and we're reviewing the eligible entries. We'll announce the winner and the pet's new name soon.

While you wait, enter /vscode-pet in chat to meet your companion and explore all its interactions and reactions.

Editor experience
Word wrap indicators
Display word wrap indicators to make wrapped lines easier to identify. An arrow at the word wrap column on the right side of the editor indicates that a line wraps.

Screenshot showing word wrap indicators in the editor.

Improved bracket auto-closing behavior
VS Code avoids inserting duplicate closing brackets when you type an opening bracket. If a matching closing bracket exists, VS Code uses it. Otherwise, VS Code inserts one.


Proposed APIs
Access token lifetime on authentication sessions
AuthenticationSession exposes an access token but no information about how long that token stays valid. An extension that passes a credential to an SDK with its own refresh callback cannot distinguish between a token that never expires and one that is about to expire. As a result, the extension either refreshes the credential unnecessarily or lets a long-running operation fail when the token expires.

The authSessionExpiration proposal adds an optional expiresAfter property to AuthenticationSession:

export interface AuthenticationSession {
  /**
   * The access token's remaining lifetime, in milliseconds, when the authentication
   * provider returns the session.
   */
  readonly expiresAfter?: number;
}

The value is the remaining lifetime when the session is returned rather than an absolute expiration timestamp. The extension host can run on a different machine than the client, and the two clocks can disagree. Authentication providers that return a cached session recompute the value each time and leave it undefined when the token's expiration is unknown. The built-in Microsoft account provider supplies this value.

Try it out and let us know what you think in the API proposal issue. To learn how to build against a proposal, see using proposed APIs.

Deprecated features and settings
Linux desktop launcher names
If a pinned launcher stops working after updating to version 1.139, remove it and add the application again from your desktop's application menu.

The Linux DEB and RPM packages now use reverse-DNS desktop file names to align with the application's desktop identity. For the Stable packages, the files in /usr/share/applications/ are renamed as follows:

Previous name	New name
code.desktop	com.microsoft.VSCode.desktop
code-url-handler.desktop	com.microsoft.VSCode.UrlHandler.desktop
Existing favorites, pinned launchers, and custom references to the old names are not updated automatically. On KDE Plasma, a stale favorite can report "You are not authorized to execute this file" even though this is a missing desktop file, not a permissions problem.

Remove the old favorite or pinned launcher, then add the application again from the application menu. On KDE Plasma, use Remove From Favorites on the old entry, then Add to Favorites on the entry under Applications > Development.
Update custom shortcuts and scripts that reference the old desktop file names.
If you explicitly configured file or URL associations using the old IDs, select the application again in your desktop's default-application settings.
You can still start the application from a terminal with code. Reinstalling the package does not update saved references to the old names.

Notable fixes
For users whose organization disables Agent mode by account policy, ensure the Welcome invitation opening is hidden and that alternative methods of launching the disabled Agents Window (for example, code --agents disallow circumvention of the control). #336968: Fix account policy enforcement in the Agents window

For users with enterprise-managed OpenTelemetry (OTel) settings, fix a race condition in the configuration of OTel in the Local (i.e. non-Agent Host Harness) to ensure OTel is not dropped. #336701: Fix Enterprise Managed OTel Race in Copilot Extension

Thank you
Contributions to vscode:

@AnupamKumar-1 (Anupam Kumar): fix(chat): preserve #file reference when editing text before it PR #333965
@baywet (Vincent Biret): feat: adds the default openapi file match for the JSON extension PR #336273
@brandonh-msft (Brandon H): Fix remote chat plugin paths PR #326916
@brignano (anthony): github-authentication: skip education.github.com check for EMU accounts PR #336608
@Chirag-Bhardwaj (Chirag Bhardwaj): Fix clearing custom terminal titles PR #336599
@dobbydobap (varshitha): Show active editor language first in Configure Snippets PR #324369
@emxs1 (Emma): add more fun working messages PR #335600
@jlelong (Jerome Lelong): Latex : update language configuration PR #332303
@joltcoke (Florian Schirmer): Guard navigator.clipboard in the WebKit clipboard workaround PR #334878
@joshspicer: Fix permission bits in mock-policy-server file commands PR #334382
@Muszic (Sangeet): Add upstream Cargo completion spec for terminal suggestions PR #305309
@SimonSiefke (Simon Siefke)
fix: memory leak in document drop edits PR #336025
fix: memory leak in dialog main service PR #336019
fix: memory leak in notebook attachment diagnostics PR #333383
fix: memory leak in signature help PR #336020
fix: memory leak in extension host hierarchy PR #336023
fix: memory leak in issueReporterOverlay PR #335102
fix: memory leak in workspace symbols PR #336029
fix: memory leak in extension host speech PR #336021
fix: memory leak in notification action view items PR #333340
@yoavbls (Yoav Balasiano): Allow display:inline-block on spans style PR #180498
Issue tracking
Contributions to our issue tracking:

@gjsjohnmurray (John Murray)
@RedCMD (RedCMD)
@IllusionMH (Andrii Dieiev)
@albertosantini (Alberto Santini)
We really appreciate people trying our new features as soon as they are ready, so check back here often and learn what's new.

If you'd like to read release notes for previous VS Code versions, go to Updates on code.visualstudio.com.

