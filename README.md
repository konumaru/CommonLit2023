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
  - 5 Fold
  - TODO: コンペの条件的にはGropuKFoldが適切
- model
  - Simgle Model
    - XGBoost
    - TODO: LightGBM
    - TODO: fine tuned deberta v3
      - inputs: text, spell_miss_count
  - averag ensemble

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
| 9 | 0.5576348008005831 | 0.478 | change kfold to group kfold |
| 10 | 0.5572727558437666 |  | add feature of spell_miss_count |
| 11 | 0.5560561772865491 | 0.479 | add feature of quotes_count |
| 13 |  |  | fine tuning deberta mdoel |

## EDA

- TODO: promptとtext間のスペルミスにどのようなものがあるか見てみる

## TODO

- bebertaのfine tuningする
- テキストの特徴量を増やす
