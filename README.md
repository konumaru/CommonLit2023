# CommonLit2023

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Kaggle Workspace is env for kaggle competition.

[Leaderboard](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/leaderboard) | [Discussion](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion?sort=published)

## Solution

todo:

- 前処理でfoldを決める
  - target encodingもこれをつかう

- feature engineering
  - TODO: text feature
    - feature list
      - text length
      - word count
      - sentence count
      - unique word count
      - most mode word count
    - method
      - auto tokenizerで処理してからカウントするとよさそう
        - <https://www.kaggle.com/code/cody11null/tuned-debertav3-lgbm-autocorrect?scriptVersionId=140573530&cellId=9>
- evaluation
  - 5 Fold
  - all oof
- model
  - text embedding
    - deberta v3
    - hogehgoe
  - GBDT
    - XGBoost
    - LightGBM
    - NN
  - averag ensemble
