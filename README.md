# Predicting Stellar Class

## プロジェクト概要

Kaggle Playground 系の多クラス分類データセットを用いて、`class` を予測するモデルを構築した。

目的変数は以下の3クラス。

- `GALAXY`
- `QSO`
- `STAR`

本プロジェクトでは、まずXGBoostを用いたベースラインを作成し、その後 `n_estimators`、`early_stopping`、`max_depth` を中心に検証を行った。

最終的には、Public LB上で最も良かった以下のモデルを現時点のベストとして採用した。

```text
XGBoost
max_depth=6
n_estimators=3000
early_stopping_rounds=100
best_iteration=1688
Public LB: 0.95579
