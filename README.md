# Predicting Stellar Class

## プロジェクト概要

Kaggle Playground 系の多クラス分類データセットを用いて、天体の種類を予測するモデルを構築した。

目的変数は `class` で、予測対象は以下の3クラス。

- GALAXY
- QSO
- STAR

本プロジェクトでは、XGBoostを用いてベースラインを作成し、その後 `n_estimators`、`early_stopping_rounds`、`max_depth` を中心に検証を行った。

現時点のベストモデルは以下。

    Model: XGBoost
    max_depth: 6
    n_estimators: 3000
    early_stopping_rounds: 100
    best_iteration: 1688
    Validation Accuracy: 0.967299
    Validation Logloss: 0.090936
    Public LB: 0.95579

---

## 使用データ

Google Drive上に配置した以下のデータを使用した。

    train: /content/drive/MyDrive/Playground (1)/train.csv
    test : /content/drive/MyDrive/Playground (1)/test.csv

---

## データ確認

最初に、以下の基本的な確認を行った。

- `train.head()` / `test.head()` によるデータ内容の確認
- `train.info()` / `test.info()` によるデータ型の確認
- `describe()` による統計量の確認
- 欠損値の確認
- 目的変数 `class` の分布確認
- 目的変数と各特徴量の関係確認
- `train` と `test` のカラム差分から目的変数を確認

今回のデータでは、`train` にのみ存在する `class` が目的変数であることを確認した。

---

## 前処理

XGBoostを使用するため、前処理は最低限に留めた。

実施した前処理は以下。

- `id` カラムを特徴量から除外
- `class` を目的変数として分離
- `LabelEncoder` で `class` を数値ラベルに変換
- `train_test_split` で学習用・検証用に分割
- `stratify=y_encoded` により、クラス比率を保ったまま分割

設定は以下。

    target_col = "class"
    id_col = "id"
    test_size = 0.2
    random_state = 42
    stratify = y_encoded

XGBoostは欠損値を内部で扱えるため、欠損値補完は行わなかった。

また、今回の特徴量は数値中心だったため、カテゴリ変数に対するOne-Hot Encodingなどの大きな前処理は不要だった。

ただし、今後 `object` 型の特徴量が存在する場合は、以下の対応が必要になる。

- One-Hot Encodingする
- `category` 型に変換して `enable_categorical=True` を使用する

---

## 評価指標

モデル評価では、主に以下を確認した。

- Train Accuracy
- Validation Accuracy
- Validation Logloss
- Train-Valid Gap
- Kaggle Public LB

ローカル検証ではValidation AccuracyとValidation Loglossを中心に見た。

ただし、KaggleではPublic LBとのズレが起きるため、最終的にはPublic LBも確認材料として用いた。

---

## 実験ログ

### 1. XGBoostベースライン

最初に、以下の設定でXGBoostのベースラインモデルを作成した。

    n_estimators = 500
    learning_rate = 0.05
    max_depth = 6
    subsample = 0.8
    colsample_bytree = 0.8
    objective = "multi:softprob"
    eval_metric = "mlogloss"
    random_state = 42

結果は以下。

    Train Accuracy: 0.969860
    Validation Accuracy: 0.964917
    Validation Logloss: 0.096486

分類レポートでは、GALAXYとQSOは高精度で分類できていた一方、STARは相対的にやや難しいクラスだった。

    GALAXY f1-score: 0.98
    QSO    f1-score: 0.96
    STAR   f1-score: 0.92

この時点で精度は高かったが、500本時点でもValidation Loglossが下がり続けていた。

そのため、`n_estimators` を増やす余地があると判断した。

---

### 2. n_estimators増加 + Early Stopping

次に、`n_estimators` を3000に増やし、`early_stopping_rounds=100` を導入した。

設定は以下。

    n_estimators = 3000
    learning_rate = 0.05
    max_depth = 6
    subsample = 0.8
    colsample_bytree = 0.8
    early_stopping_rounds = 100
    tree_method = "hist"

結果は以下。

    best_iteration: 1688
    best_score: 0.090936
    Train Accuracy: 0.980880
    Validation Accuracy: 0.967299
    Validation Logloss: 0.090936
    Public LB: 0.95579

ベースラインからの改善は以下。

| Model | Train Accuracy | Validation Accuracy | Validation Logloss | Public LB |
|---|---:|---:|---:|---:|
| Baseline XGBoost | 0.969860 | 0.964917 | 0.096486 | 0.95255 |
| XGBoost + Early Stopping | 0.980880 | 0.967299 | 0.090936 | 0.95579 |

Validation AccuracyとValidation Loglossがどちらも改善した。

一方で、Train Accuracyも上昇しており、Train-Valid Gapは広がった。

    Train Accuracy: 0.980880
    Validation Accuracy: 0.967299
    Gap: 約1.36pt

若干過学習気味ではあるが、Validation性能とPublic LBが改善したため、このモデルは有効だった。

---

### 3. max_depthの調整

次に、過学習を抑える目的で `max_depth` を調整した。

当初は以下のGridSearchを検討した。

    max_depth: [5, 6, 7, 8]
    min_child_weight: [1, 3, 5]

しかし、CPU実行では非常に時間がかかったため、探索範囲を縮小した。

その後、ColabのランタイムをT4 GPUに変更し、XGBoostに `device="cuda"` を指定した。

これにより、1モデルあたりの学習時間が大幅に短縮された。

CPU実行では1モデルあたり数分以上かかっていたが、T4 GPUでは1モデルあたり数十秒程度で学習できた。

---

### 4. max_depth=5モデル

`max_depth=6` がやや過学習気味だったため、木の深さを浅くした `max_depth=5` のモデルを作成した。

設定は以下。

    n_estimators = 4000
    learning_rate = 0.05
    max_depth = 5
    subsample = 0.8
    colsample_bytree = 0.8
    objective = "multi:softprob"
    eval_metric = "mlogloss"
    early_stopping_rounds = 100
    tree_method = "hist"
    device = "cuda"

結果は以下。

    best_iteration: 2495
    best_score: 0.091053
    Train Accuracy: 0.978377
    Validation Accuracy: 0.967238
    Validation Logloss: 0.091053
    Public LB: 0.95560

`max_depth=6` と比較すると、Train Accuracyは下がった。

    max_depth=6 Train Accuracy: 0.980880
    max_depth=5 Train Accuracy: 0.978377

そのため、過学習はやや抑えられたと考えられる。

一方で、Validation Accuracy、Validation Logloss、Public LBはいずれもわずかに悪化した。

    max_depth=6 Validation Accuracy: 0.967299
    max_depth=5 Validation Accuracy: 0.967238

    max_depth=6 Validation Logloss: 0.090936
    max_depth=5 Validation Logloss: 0.091053

    max_depth=6 Public LB: 0.95579
    max_depth=5 Public LB: 0.95560

このため、今回のデータでは `max_depth=5` は少し保守的すぎた可能性がある。

---

## 提出結果まとめ

| No. | Model | 主な設定 | Validation Accuracy | Validation Logloss | Public LB |
|---:|---|---|---:|---:|---:|
| 1 | XGBoost Baseline | n_estimators=500, max_depth=6 | 0.964917 | 0.096486 | 0.95255 |
| 2 | XGBoost + Early Stopping | n_estimators=3000, max_depth=6, best_iteration=1688 | 0.967299 | 0.090936 | 0.95579 |
| 3 | XGBoost depth5 | n_estimators=4000, max_depth=5, best_iteration=2495 | 0.967238 | 0.091053 | 0.95560 |

現時点のベストは、`max_depth=6` のEarly Stoppingモデル。

    Best Model:
    XGBoost
    max_depth = 6
    n_estimators = 3000
    early_stopping_rounds = 100
    best_iteration = 1688
    Public LB = 0.95579

---

## 試した仮説と結果

### 仮説1：n_estimatorsを増やせば性能が上がる

ベースラインでは、`n_estimators=500` 時点でもValidation Loglossが下がり続けていた。

そのため、木の本数を増やすことで、まだ性能が伸びると考えた。

結果として、`n_estimators=3000` に増やし、Early Stoppingを導入することで、Validation AccuracyとValidation Loglossが改善した。

    Validation Accuracy: 0.964917 -> 0.967299
    Validation Logloss: 0.096486 -> 0.090936

この仮説は当たりだった。

---

### 仮説2：Early Stoppingを使えば、過学習を抑えながら木の本数を増やせる

`n_estimators` を大きくすると、過学習のリスクがある。

そこで、`early_stopping_rounds=100` を導入し、Validation Loglossが改善しなくなった時点で学習を止めるようにした。

結果として、3000本すべてを使うのではなく、`best_iteration=1688` で停止した。

    n_estimators: 3000
    best_iteration: 1688

これにより、木の本数を増やしつつ、過学習をある程度コントロールできた。

この仮説も有効だった。

---

### 仮説3：max_depthを浅くすれば過学習が抑えられ、Public LBが改善する

`max_depth=6` では、Train AccuracyとValidation Accuracyの差がやや広がっていた。

そこで、`max_depth=5` に下げることで、モデルの複雑さを抑え、Public LBが改善するのではないかと考えた。

結果として、Train Accuracyは下がった。

    max_depth=6 Train Accuracy: 0.980880
    max_depth=5 Train Accuracy: 0.978377

しかし、Validation性能とPublic LBは改善しなかった。

    max_depth=6 Public LB: 0.95579
    max_depth=5 Public LB: 0.95560

このため、今回のデータでは `max_depth=5` はやや表現力不足だった可能性がある。

過学習を抑えること自体は重要だが、単純に浅くすれば良いわけではないと分かった。

---

### 仮説4：Validation Accuracyが高ければPublic LBも改善する

今回、Validation Accuracyはモデル選択の参考になった。

ただし、Validation Accuracyが近いモデルでもPublic LBには差が出た。

特に、`max_depth=5` はValidation Accuracy上では `max_depth=6` とかなり近かったが、Public LBではやや下がった。

    max_depth=6 Validation Accuracy: 0.967299
    max_depth=5 Validation Accuracy: 0.967238

    max_depth=6 Public LB: 0.95579
    max_depth=5 Public LB: 0.95560

このことから、Validationだけで判断するのではなく、Public LBとの対応も確認する必要があると分かった。

---

### 仮説5：GPUを使えば探索効率が大きく上がる

CPU実行では、XGBoostのGridSearchに非常に時間がかかった。

T4 GPUに変更し、`device="cuda"` を指定すると、学習時間が大幅に短縮された。

特に、`max_depth=5` のモデルは以下の時間で学習できた。

    elapsed: 約38.4秒

今後、XGBoostで複数パラメータを探索する場合は、GPU利用を前提にした方が良い。

---

## 今日の学び

### 1. XGBoostでは、n_estimatorsを大きくしてEarly Stoppingするのが有効

最初から木の本数を小さく固定するより、大きめに設定してEarly Stoppingで止める方が良い。

今回も、500本ではまだValidation Loglossが下がり続けていたため、3000本まで増やすことで性能が改善した。

---

### 2. best_iterationはパラメータごとに変わる

`n_estimators=3000` や `n_estimators=4000` は、あくまで最大本数である。

実際に使われる木の本数は、Early Stoppingによって決まる。

今回の例では以下のようになった。

    max_depth=6 -> best_iteration=1688
    max_depth=5 -> best_iteration=2495

浅い木は1本あたりの表現力が弱いため、より多くの木が必要になる傾向がある。

---

### 3. 過学習を抑えれば必ずPublic LBが上がるわけではない

`max_depth=5` に下げることで、Train Accuracyは下がり、過学習はやや抑えられた。

しかし、Public LBは改善しなかった。

つまり、過学習を抑えること自体は重要だが、モデルの表現力を落としすぎるとスコアが下がる。

今回のデータでは、`max_depth=6` 程度の複雑さが必要だった可能性が高い。

---

### 4. ValidationとPublic LBは完全には一致しない

Validation AccuracyやValidation Loglossは重要だが、それだけでPublic LBの結果を完全には予測できない。

今回も、Validation上ではかなり近いモデル同士で、Public LBに差が出た。

そのため、基本はValidationを主軸にしつつ、Public LBは確認材料として使うのが良い。

ただし、Public LBに過剰適合しすぎないよう注意する必要がある。

---

### 5. GridSearchは探索範囲を絞ることが重要

最初から広い探索範囲を設定すると、計算時間が非常に重くなる。

当初検討した探索範囲は以下だった。

    max_depth: [5, 6, 7, 8]
    min_child_weight: [1, 3, 5]

これは12通りの探索になるため、CPUではかなり重かった。

まずは `max_depth` のように重要なパラメータを1つずつ動かし、方向性を見てから探索を広げる方が効率的だと分かった。

---

## 現時点の結論

今回の範囲では、最も良かったモデルは以下。

    XGBoost
    max_depth = 6
    n_estimators = 3000
    early_stopping_rounds = 100
    best_iteration = 1688
    Validation Accuracy = 0.967299
    Validation Logloss = 0.090936
    Public LB = 0.95579

`max_depth=5` は過学習を少し抑えたものの、Validation LoglossとPublic LBでは改善しなかった。

そのため、現時点では `max_depth=6` のEarly Stoppingモデルをベストモデルとして採用する。

---

## 今後試すなら

### 1. min_child_weightの再探索

今回は時間の都合で `min_child_weight` を本格的には試し切れなかった。

`max_depth=6` を固定した上で、以下を試す価値がある。

    min_child_weight: [1, 3, 5, 7]

`max_depth=6` の表現力を維持しながら、過学習を少し抑えられる可能性がある。

---

### 2. subsample / colsample_bytreeの調整

現在はどちらも `0.8` に固定している。

過学習を抑える方向では、以下の範囲を試したい。

    subsample: [0.7, 0.8, 0.9, 1.0]
    colsample_bytree: [0.7, 0.8, 0.9, 1.0]

行方向・列方向のサンプリングを調整することで、汎化性能が改善する可能性がある。

---

### 3. learning_rateを下げてn_estimatorsを増やす

現在は `learning_rate=0.05`。

より丁寧に学習するなら、以下の設定も試す価値がある。

    learning_rate = 0.03
    n_estimators = 5000
    early_stopping_rounds = 150

ただし、学習時間は長くなるため、GPU使用が前提になる。

---

### 4. Stratified KFoldによるCross Validation

今回は単一のtrain/valid splitで評価した。

より安定した検証をするなら、Stratified KFoldを使ったCVを導入したい。

単一のValidationに依存しすぎると、分割の偶然に影響される可能性がある。

CV平均で評価することで、Public LBとのズレを少し減らせる可能性がある。

---

### 5. アンサンブル

単体モデルの改善が頭打ちになってきた場合、以下のアンサンブルも検討できる。

- max_depth違いのXGBoostを平均
- seed違いのXGBoostを平均
- XGBoost / LightGBM / CatBoost の平均
- VotingまたはSoft Voting

ただし、まずは単体モデルでの検証を十分に行ってから取り組む。

---

## 最終メモ

今日の一番大きな学びは、以下の3つ。

1. `n_estimators` を増やしてEarly Stoppingするのは有効
2. 過学習を抑えれば必ずPublic LBが上がるわけではない
3. ValidationとPublic LBのズレを見ながら、小さく仮説検証することが重要

今回のプロジェクトでは、単にスコアを追うだけではなく、以下を意識して検証した。

- なぜそのパラメータを試すのか
- Validationでは何が改善したのか
- Public LBではどうだったのか
- 過学習は増えたのか減ったのか
- 次に試すべき仮説は何か

一旦、現時点では以下をベストモデルとして区切りを打つ。

    Best Model:
    XGBoost
    max_depth = 6
    n_estimators = 3000
    early_stopping_rounds = 100
    best_iteration = 1688
    Public LB = 0.95579
