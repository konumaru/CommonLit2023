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
    - TODO: quotes overlap count
  - promtp & text embedding
    - deberta v3
  - TODO: cosine similarity of prompt & text embeddings
  - TODO: target encoding of content and wording
- cv strategy
  - 5 Fold (promtp_idでGroupKFoldする？)
- model
  - Simgle Model
    - XGBoost
    - TODO: LightGBM
    - TODO: NN
  - averag ensemble

## Experiments

| EXP_ID | Local CV | Public LB | Note |
| :---: | :---: | :---: | :--- |
| 1 | 0.6687954845101823 | 0.599 | rf with simple text feature |
| 2 | 0.5148155805419965 | -- | add debertav3 text embeddings feature |
| 3 | 0.4903529269444470 | 0.509 | change model from rf to xgb |
| 4 | 0.4899955738087213 | -- | add debertav3 prompt embeddings feature |
| 5 | 0.4785185756657641 | -- | add overlap word and co-occur words feature |
| 6 | 0.4759433370779221 | -- | add tri-gram co-occur words feature |
| 7 | 0.4737618975123431 | -- | change xgb n_estimatoers param 500 to 800 |

## EDA

- TODO: promptとtext間のスペルミスにどのようなものがあるか見てみる
