# CommonLit2023

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Leaderboard](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/leaderboard) | [Discussion](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion?sort=published)

## Solution

- feature engineering
  - feature list
    - text length
    - word count
    - sentence count
    - quoted sentence count
    - consec tive_dots count
    - word overlap count
    - spell miss count, SpellChecker
  - text embedding
    - deberta v3
  - TODO: target encoding of content and wording
    - count系の特徴量を使う？
- cv strategy
  - Group K-Fold
    - k: 4
    - group:  prompt_id
- model
  - first statge model
    - fine tuned deberta v3
      - inputs: text
  - second stage models
    - XGBoost (cv=0.5168956770838019)
      - inputs: first stage output, text feature
    - LightGBM (cv=0.5178261265275084)
      - inputs: first stage output, text feature
  - ensemble
    - simple average of second stage models.

## Experiments

| EXP_ID | Local CV | Public LB | Note |
| :---: | :---: | :---: | :--- |
| 1 | 0.6687954845101823 | 0.599 | rf with simple text feature |
| 2 | 0.5148155805419965 | -- | add feature of debertav3 text embeddings |
| 3 | 0.4903529269444470 | 0.509 | change model from rf to xgb |
| 4 | 0.4899955738087213 | -- | add featrue of debertav3 prompt embeddings |
| 5 | 0.4785185756657641 | -- | add feature of overlap word and co-occur words |
| 6 | 0.4759433370779221 | -- | add feature of tri-gram co-occur words |
| 7 | 0.4737618975123431 | -- | change xgb n_estimatoers param 500 to 800 |
| 8 | 0.4744999729694380 | 0.479 | rm featrue of debertav3 prompt embeddings |
| - | ------------------ | -- | change evaluate mothod |
| 9 | 0.5576348008005831 | 0.478 | change kfold to group kfold |
| 10 | 0.5572727558437666 |  | add feature of spell_miss_count |
| 11 | 0.5560561772865491 | 0.479 | add feature of quotes_count |
| 12 | 0.5451717268584183 | 0.559 | only finetuned deberta base |
| 13 | 0.5168956770838019 | 0.491 | stacking xgb on deberta |
---
| 14 |  |  | ensenble lgbm |
| 15 |  |  | add feature of target encoding |

## Not Worked For Me

- Fine tune with content and wording at the same.
