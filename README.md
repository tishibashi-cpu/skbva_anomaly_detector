# 圧力異常検知 監視ダッシュボード — 利用マニュアル

SuperKEKB 真空グループ

末次さんの圧力異常検知プログラムの判定結果を、ブラウザのダッシュボードで見られるようにした一式です。
判定の中身（末次さんのプログラム）には手を入れず、表示と運用まわりだけを足しています。

このマニュアルは大きく2部構成です。

- **第I部：メンバー向け** … ブラウザでダッシュボードを見るだけの人はここだけ読めば十分です。
- **第II部：管理者向け** … サーバー側でプログラムを動かす担当者向けの設定・運用です。

> このマニュアル内の **`your_account`** は、各自の KEK ログイン名に読み替えてください。

---

# 第I部：メンバー向け

## 1. ダッシュボードをブラウザで見る（いちばんよく使う手順）

ダッシュボードのサーバーは、グループで **1台**だけ起動しておけば十分です（誰か一人＝管理者が動かします）。
各メンバーは自分の PC から **SSH トンネル**を1本張って、ブラウザで見るだけです。

### 手順A：初回だけ、SSH の設定を1回書いておく

自分の PC の `~/.ssh/config` に次を追記します（`your_account` は自分のログイン名に置換）。

```
Host ckdash
    HostName kekb-co-user01
    User your_account
    ProxyJump your_account@kekb-login1
    LocalForward 18050 localhost:18050
```

> `HostName` は、管理者がダッシュボードを起動しているホスト名に合わせてください（通常 `kekb-co-user01`）。
> 起動ホストが違うと、トンネルは張れても画面が出ません。

### 手順B：見るたびにやること

1. 自分の PC のターミナルで、トンネルを張る（パスワードを2回聞かれます）。

   ```
   ssh ckdash
   ```
   つながったら、**このターミナルは開いたまま**にしておきます（閉じるとトンネルが切れます）。

2. 自分の PC の **ブラウザ**で次を開く。

   ```
   http://localhost:18050
   ```

これでダッシュボードが表示されます。画面は5秒ごとに自動更新されます。
見終わったら、ブラウザを閉じてターミナルで `Ctrl-C`（またはターミナルを閉じる）でトンネルを切ります。

> `~/.ssh/config` を使わない一行版（毎回手で打つ場合）:
> ```
> ssh -J your_account@kekb-login1 -L 18050:localhost:18050 your_account@kekb-co-user01
> ```

### うまく表示されないとき

| 症状 | 原因と対処 |
|------|-----------|
| ブラウザに「接続できません / Connection refused」 | トンネルを張っていない、または管理者がダッシュボードを起動していない。手順B-1 を実行し、それでもダメなら管理者に起動を依頼。 |
| `channel ... connect failed: Connection refused`（ターミナル側） | 同上。トンネル先（サーバー）でダッシュボードが起動していない。 |
| 画面に「更新失敗」と出る | サーバー側でダッシュボードが止まった可能性。トンネルを張り直し、必要なら管理者に確認。 |
| `bind: Address already in use`（トンネルを張ろうとして） | 自分の PC の 18050 番が他で使用中。設定の左側だけ変える（例 `LocalForward 28050 localhost:18050`）→ ブラウザは `http://localhost:28050`。 |

---

## 2. このプログラムは何をしているのか

判定の考え方を、用語をできるだけ使わずにまとめます（詳しくは末尾「16. 参考文献」の論文）。

### 2-1. 何のためのもの？

加速器のリング（LER・HER）には、真空の良し悪しを測る**真空計（CCG）が約600台**並んでいます。
リークや異常な放出ガス、放電などが起きると、その場所の**圧力が異常に上がります**。
600台を人間が四六時中見張るのは大変なので、このプログラムが**自動でチェック**して、
「いつもと違う動きをしている真空計」をダッシュボードに挙げてくれます。

### 2-2. どうやって「異常」と判断している？

ポイントは、**「決まった圧力の値を超えたら異常」ではない**ことです。
そうではなく、**「その真空計自身の“少し前のふつうの振る舞い”と比べて、今が大きくズレているか」**で判断します。

イメージとしては、各真空計について次をやっています。

1. **ふつうの姿を学ぶ**：数日前のデータ（基準データ）を見て、「ビーム電流がこのくらいなら圧力はこのくらい」
   という、その真空計にとって正常な関係（曲線）を求める。
2. **今と比べる**：直近のデータ（調査データ）が、その正常な曲線からどれだけズレているかを測る。
3. **ズレが大きければ異常**としてダッシュボードに挙げる。

つまり「絶対的な圧力の高さ」ではなく「**最近の自分と比べた変化の大きさ**」を見ています。

### 2-3. 機械学習はどこで使われている？

ズレの大きさから「正常／異常」を最終判定する部分と、「異常の原因はおそらく何か」を推定する部分に、
**あらかじめ学習させた小さなニューラルネットワーク**を使っています。

- この学習は**過去（2016〜2024年）の実際の真空トラブル約20例**から**一度だけ**行い、結果はファイル
  （`model_*.h5` など）に保存済みです。
- **動かしている間に勝手に学習し直すことはありません**。毎回やっているのは「保存済みの判定基準にあてはめる」だけです。
- 推定原因は「リーク or ポンプ故障」「異常加熱 or 放電」「軌道異常 or リーク」「圧力バースト」などに分類されます。

### 2-4. 「カウント」と深刻度

チェックはビームアボート（フィルの区切り）ごとに行われます。
ある真空計が**直近8回のチェックのうち何回 異常と判定されたか**を数え、その回数で深刻度を色分けします。

- **要注意（赤）**：8回中 6回以上が異常
- **注意（黄）**：3〜5回が異常
- それ未満は表示しません。

### 2-5. 知っておきたい注意点（重要）

判断が「最近の自分との比較」であるため、**異常な高い圧力が長く続いて“定常化”すると、
いずれ正常判定に戻ることがあります**。

理由は、時間が経つと「基準データ（数日前）」のほうもその高い圧力状態になり、「今」と「基準」が
似てきてズレが小さくなるからです。システムはそれを「新しいふつう」とみなしてしまいます。

したがって、このダッシュボードは**「急な異常・進行中の異常」を見つけるのが得意**な一方、
**ゆっくりした変化や高止まりした異常は見逃しうる**という性質があります。
**日々の圧力値の目視点検と併用**してください（高止まりは目視点検で気づけます）。

### 2-6. 圧力（CCG）以外にも4つの判定軸がある

ここまでの説明は「圧力異常検知」タブ（CCG約600台）についてでした。ダッシュボードには
他に4つのタブがあり、それぞれ独立した判定を行っています（見方は次章）。

| タブ | 見ているもの | 判定の方式（ざっくり） |
|---|---|---|
| イオンポンプ異常放電 | イオンポンプの放電電流の異常 | 正常期間に学習した圧力-電流関係（固定モデル）とのズレ |
| 温度計異常検知 | ビームパイプ温度計**センサ自体**の故障 | CCGと同じ「数日前の自分」とのローリング比較 |
| 機器劣化検知 | センサは正常な前提で、**測定対象機器側**の熱結合劣化（放熱不良等） | 運用者が明示的に学習した過去の健全期間（固定モデル）との比較。年単位の緩慢な変化向け |
| 流量計異常検知 | 冷却水流量計**センサ自体**の故障（実際の流量低下は別のアラームが担当） | 学習不要。直近の指示値だけを固定閾値で判定 |

いずれも「異常の候補を挙げて人間の確認を促す」もので、最終判断は現場での確認が前提です。

---

## 3. ダッシュボードの見方

画面上部に **タブが5つ**あります：「圧力異常検知」「イオンポンプ異常放電」「温度計異常検知」
「機器劣化検知」「流量計異常検知」。クリックで切り替えます（同じページ内でJSでの切り替え。
ページ遷移はしません）。各タブの見出しにその日の異常件数バッジが出ます（sev3を含む場合は赤）。

> **黄色いバナー「⚠ 現在、アーカイバ(kblog)が…データ取得を停止中です」について**：
> シャットダウン期間などでアーカイバが温度計・流量計のデータ記録を止めている間は、
> 温度計異常検知・機器劣化検知・流量計異常検知の各タブにこのバナーが出て、タブのバッジも
> 「停止中」になります。これは**センサや機器の異常ではなく**、判定に使うデータ自体が
> 取得できていない状態を示すものです（アーカイバが復旧すれば自動的に通常の判定表示に戻ります）。

各部の意味です。

**上部のヘッダ**（全タブ共通）
- **LER/HER のビーム状態**：2つの値を並べて表示します。
  - **現在 … mA**：蓄積電流 PV（`BMLDCCT:CURRENT` / `BMHDCCT:CURRENT`）から数秒ごとに取る**リアルタイム値**。
  - **最終チェック …**：検知プログラムが**最後に判定したとき**のビーム状態（検知由来）。
  入射直後などは「現在 50 mA ⋅ 最終チェック OFF」のように、リアルタイムは流れているのに最終チェックは
  まだ前回（無ビーム時）のまま、という表示になります。次にアボートか定期チェックが走れば最終チェック側も更新されます。
  （EPICS が読めない環境ではリアルタイム表示は省かれ、最終チェックのみ表示）
- **負荷 / CPU / Mem**：ダッシュボードを動かしている**共用サーバー全体**の混み具合（自分専用の数字ではありません）。
  コア数に対する比で色が変わります（緑＝余裕／橙＝やや高い／赤＝コア数超過）。
- **本プログラム CPU / Mem**：このダッシュボード自身が使っている分だけの数字。
- **検知プログラム CPU / Mem**：`detector_headless.py`（`--watch`等）の使用量。ダッシュボードとは
  親子関係の無い**別プロセス**なので、上の「本プログラム」には一切含まれない（実機で
  `--watch`実行中にCPU使用率が1000%超に達した際、「本プログラム」側の表示では気づけなかった
  ため追加）。`.detector.lock`からPIDを教えてもらって計測しており、`--watch`が動いていない
  ときは「検知プログラム: 停止中/未検出」と表示される。CPU%はサーバー全体に対する割合なので、
  左の全体CPU%と直接比較できる（目安: 1プロセスで20%以上なら黄色、50%以上なら赤色）。
- **最終チェック**：検知プログラムが最後に判定した時刻。
- **判定: FNN**：判定に使っている方式（ニューラルネット）。

**「圧力異常検知」タブ**

**サマリーカード**
- **監視中の真空計**：見張っている台数（例 605、内訳 LER 308 / HER 297）。
- **要注意 (≥6) / 注意 (3–5)**：直近で挙がっている真空計の件数。
- **状態**：要注意が1件でもあれば ALERT、なければ OK。

**検知された異常（カウントの多い順）**
- 真空計名・リング・側室（D01〜D12）・モード（Storage＝蓄積中／Tail＝アボート直後）。
- **推定: …**：推定される原因。
- 右の小さなプロット：そのレコードの **異常カウントの推移**。縦軸 **Anomaly Count**（＝直近 `max_count`
  フィル窓で異常と出た回数 0〜max_count）、横軸 **Period**（フィル順, 古い→新しい）、右軸に **Beam [mA]**。
  異常の進行が見える（フィル要約が無いと圧力スパークラインにフォールバック）。
- 行をクリックすると詳細グラフが開きます：**圧力 vs ビーム電流の散布図**（実測点＋調査回帰／
  基準回帰の2入力モデル `圧力=w0·I+w1·(I²/Nb)²+w2` 曲線）と、**圧力＆ビーム電流 vs 時刻**の2軸プロット。
- 異常が無い静かな期間は「**直近N日に検知された異常はありません ✓**」と表示され、
  参考までに「最後に記録された異常: …」が添えられます。

**「イオンポンプ異常放電」タブ**
- 放電電流の異常を検知したイオンポンプを一覧します（急性を上に表示）。
- 各カードに深刻度バッジ（**sev3/sev2/sev1 すべて表示**）・急性/慢性バッジ・**「N 回連続」バッジ**・
  逸脱量（+X.X dex）を表示し、右側に **異常カウントの推移プロット**（縦軸 **Anomaly Count**＝sev3
  連続カウント、横軸 **Judge Cycle**）を描く。
- クリックすると詳細グラフが開きます：**放電電流 vs 時刻**（無ビーム区間を網掛け・学習バンド
  p50–p95 を水平帯で重ね描き）と、**電流 vs 圧力**（I-P・両対数、`I=a·P^b` 回帰つき）。

**「温度計異常検知」タブ**
- LER/HER それぞれについて、直近窓（既定24h）で判定した異常をランク表示します（`temp_detector/`
  が生成する `temp_dashboard_state.json` を読む。無ければ「temp_headless.py を実行してください」
  という案内が出るだけで、他タブの動作には影響しません）。
- 列：severityバッジ（sev1〜3）・PV・種別（BL/CLM/GV等）・t_med/t_min/t_max・理由（短絡疑い／
  無ビーム高温／ビーム反相関／グリッチ頻発 等、日本語）。
- 行をクリックすると詳細（section/tag・持続割合・ビーム相関等の内部メトリクス）と、
  **温度＆ビーム電流 vs 時刻**のグラフ（左軸：温度[℃]・赤、右軸：ビーム電流[mA]・緑点線。
  判定窓の間引き時系列）が展開されます。

**「機器劣化検知」タブ**
- 温度計**センサ自体**の故障を見る「温度計異常検知」タブとは別の判定軸で、**センサは正常な前提で、
  測定対象の機器側**（放熱不良・断熱劣化・接触不良による発熱増加等）の劣化を見ます（IR/LER/HER
  いずれのリングも対象）。`temp_detector/` が生成する `temp_equipment_state.json` を読む
  （`detector_headless.py --equipment-judge` または相乗り実行、もしくは `temp_equipment.py
  judge-all` で更新）。
- リングごとに、まだ `learn`（基準期間の学習）していなければ「未学習のためスキップ」と表示され、
  他のリングやタブの動作には影響しません。
- 列：severityバッジ（sev1〜3）・PV・モデル種別（`linear`/`hom`）・比・基準/現在の値（`linear`は
  傾き dT/dI、`hom`は代表運転電流での予測発熱量）・理由（発熱増加(軽度/中/重度) 等、日本語）。
- 行をクリックすると詳細（section/tag・環境温度差(Δa/Δw0、参考)・フィット品質等）と、
  **温度 vs ビーム電流の散布図**（基準期間＝薄青・調査期間＝赤の実測点と、それぞれのフィット
  曲線。ビームあり点のみ）が展開されます。`hom`型は判定のために基準期間の生データも再取得する
  ので基準側の散布点も表示されますが、`linear`型は保存済みフィット係数しか持たないため基準は
  フィット直線のみです。
- 機器の熱結合特性は年〜数年スケールでしか動かない量のため、他タブと違って**判定は毎回学習し
  直さず**、運用者が明示的に学習した固定モデルを使い続けます（詳細は第II部 14 章「機器劣化検知」節）。

**「流量計異常検知」タブ**
- 冷却水流量計の**センサ自身の異常**だけを見ます（実際の流量低下は別のアラームシステムが検知
  するため対象外。運用上、流量計は分解清掃すれば指示値が元に戻る個体故障がほとんどで、実際に
  流量が落ちたことは無いとのこと）。ビーム電流と無関係の機器なので、CCG/温度計/機器劣化検知と
  違い**リング(LER/HER)の概念を持たず**、直近窓（既定24h）の指示値だけで判定します
  （`flow_detector/` が生成する `flow_dashboard_state.json` を読む）。
- 列：severityバッジ（sev1〜3）・PV・セクション（D01〜D12）・校正基準比[%]（100%＝ある時点で
  取った基準流量と同じ流量）・CV[%]（指示値の変動係数）・理由（値の固着／校正基準比で低下／
  指示値不安定 等、日本語）。
- 行をクリックすると詳細（tag/sensor_id・有効点数・張り付き検知時のrange・外れ値の本数）と、
  **流量 vs 時刻**のグラフ（縦軸：流量[%]。判定窓の間引き時系列）が展開されます。
- 詳細は第II部 15 章「拡張：冷却水流量計異常検知」節を参照。

---

# 第II部：管理者向け（サーバーでプログラムを動かす人）

ここから先は、検知プログラムとダッシュボードをサーバー上で動かす担当者向けです。
メンバーとして見るだけなら読む必要はありません。

## 4. 全体構成

```
検知（判定）            → 結果ファイル → アダプタ          → 表示
detector_headless.py      *_Result_*    state_builder.py   dashboard.py
（末次プログラムを GUI なしで              → dashboard_state.json → ブラウザ:18050（5タブ）
  import して判定を回す。IP/温度計/機器劣化/流量計も相乗り）
```

新しいプログラム群は **トップ（`~/skbva_anomaly_detector`）** に置き、末次さんのプログラム本体と実行時データは
**`legacy/`** に、温度計異常検知は独立した自己完結パッケージとして **`temp_detector/`** にまとめています。

```
~/skbva_anomaly_detector/                            ← ここから各プログラムを起動
├── detector_headless.py             検知本体を GUI なしで回す（実機: EPICS/kblogrd 必要）。
│                                     CCG判定に加え、IP judge・温度計judgeも段階分散して相乗り実行する
├── state_builder.py                 legacy/ の結果ファイル → dashboard_state.json
├── dashboard.py                     JSON をブラウザ配信（:18050）。5タブ（圧力/イオンポンプ/温度計/機器劣化/流量計）。
│                                     Save Normal/Abnormal ボタンは label_queue.jsonl に追記。
│                                     --port / DASHBOARD_PORT で別ポート起動可（本番と共存できる）
├── apply_labels.py                  label_queue.jsonl を読み legacy の Save_Manual_* で教師行を追記（実機・人手実行）
├── label_queue.jsonl                教師ラベルのキュー（ボタン押下で追記。実行時生成）
├── record_raw.py                    詳細ビュー用の生データ取得。CCG=2入力モデル 圧力=w0·I+w1·(I²/Nb)²+w2 の散布図/時系列、IP=電流vs時刻・I-P散布図（学習バンド/I=a·P^b 重ね描き）
├── ccg_pv.py                        監視CCGリストを CSV から読む
├── cause_infer.py                   推定原因をヘッドレス再現（純 numpy、h5py のみ）
├── sysload.py                       サーバー負荷＋自プロセス使用量（標準ライブラリのみ）
├── beamcurrent.py                   蓄積電流（リアルタイム）を EPICS から読む
├── singleton.py                     二重起動防止の PID ロック（標準ライブラリのみ）。ロックファイルは
│                                     ポートごとに分ける（`.dashboard.<port>.lock`）ので、本番と別ポートの
│                                     デモ起動を同時に走らせても衝突しない
├── ip_pv.py                         イオンポンプPV読み込み（電源種別判別）※拡張
├── ip_fetch.py                      イオンポンプ放電電流の履歴を kblogrd で取得 ※拡張
├── ip_state.py                      取得履歴→ダッシュボード用の ion_pumps 構造に変換（現在は renderIonPumps
│                                     が no-op のため非表示。judge セクションに一本化）※拡張
├── ccg_fetch.py                     CCG 圧力の履歴を kblogrd で取得（観察用）※拡張
├── ip_observe.py                    同一地点ペアの圧力 vs 放電電流を PNG 出力（観察用）※拡張
├── ip_judge.py                      イオンポンプ放電電流の異常判定（学習＋L0a/L0b/L1/L2＋ビーム軸格下げ＋急性/慢性）※拡張（運用中）
├── beam_fetch.py                    蓄積ビーム電流の履歴を kblogrd で取得（判定用）※拡張
├── ip_data.json                     ip_state が保存、state_builder が読む ※拡張
├── ip_models.json                   ip_judge learn が保存、judge が読む（固定モデル方式。実行時生成）※拡張
├── ip_models_rolling.json           IP_JUDGE_ROLLING=True 時のみ。judge のたびに直近数日から作り直す
│                                     ローリング基準モデル（実行時生成、既定では未使用）※拡張
├── ip_judge_state.json              judge --out-json の出力。state_builder が読み専用セクションに反映（実行時生成）※拡張
├── ip_judge_counts.json             sev3 の累積カウント（継続したものだけ表示するため。実行時生成）※拡張
├── ip_judge_history.json            PVごとの sev3 連続カウント履歴（カードの推移プロット用。実行時生成）※拡張
├── tools/                           調査用スクリプト（band_check / beam_peek / ip_corr_survey / ip_beam_survey）。判定本体には不要
├── pv_info/                         PV リストの CSV をまとめて置く（CCG/IP/温度計 共通の置き場）
│   ├── LER_CCG_PV.csv / HER_CCG_PV.csv       監視対象 CCG（1列目 PV名のみ）
│   ├── LER_IP_PV.csv / HER_IP_PV.csv         監視対象イオンポンプ電流（1列目 PV名）※拡張
│   ├── LER_TEMP_PV.csv / HER_TEMP_PV.csv     監視対象ビームパイプ温度計（VA{L,H}TMP形式）※拡張
│   ├── IR_TEMP_PV.csv                        IR（衝突点周辺）フォーカス磁石温度計（FB_MOVE形式）※拡張
│   ├── TEMP_RING_OVERRIDE.csv                温度計のリング所属の例外（配線間違い・一時移設等。手編集可）※拡張
│   └── FLOW_PV.csv                           監視対象冷却水流量計（1列目 PV名のみ。リング分けは無い）※拡張
├── temp_detector/                   温度計異常検知（自己完結パッケージ。CCG/IPのコードに依存しない）※拡張
│   ├── temp_pv.py                       PV名パーサ（VA{L,H}TMP形式＋IR/FB_MOVE形式。上下ペア検出）
│   ├── temp_fetch.py                    kblogrd取得（VA/VATemp・IRはBM/BMOthers）、ビーム/バンチ数(Misc/Base)取得、リング例外の適用
│   ├── temp_judge.py                    判定コア（センサ故障。層: H0範囲/H1張り付き/N ノイズ/G グリッチ/S 短絡/
│   │                                     O 無ビーム高温/B ビーム反相関/I 間欠逸脱/P 上下ペア）
│   ├── temp_batch.py                    全台バッチ判定CLI（`run`に`--rolling`でCCG式ローリング基準）
│   ├── temp_headless.py                 定期実行（cron `--once` or 常駐ループ、両リング判定して
│   │                                     temp_dashboard_state.json を書く）
│   ├── temp_probe.py                    特定センサの波形・判定を見る較正/診断用ツール（センサ故障側）
│   ├── temp_equipment.py                機器劣化検知（learn/judge/judge-all/compare/scan。linear/homモデル、
│   │                                     全リング対応。temp_equipment_models.json を生成）
│   └── temp_equipment_plot.py           機器劣化検知の比較散布図（温度 vs ビーム電流）を出力
├── temp_detector/temp_equipment_models.json   temp_equipment.py learn が保存、judge/run_periodic_judge が読む
│                                     （固定モデル方式。実行時生成）※拡張
├── temp_detector/temp_equipment_state.json    run_periodic_judge（detector_headless.py 相乗り／
│                                     judge-all）が書く。dashboard.py の「機器劣化検知」タブが読む（実行時生成）※拡張
├── flow_detector/                   冷却水流量計異常検知（自己完結パッケージ。ビーム電流と無関係・
│                                     リング概念も持たない。他パッケージに依存しない）※拡張
│   ├── flow_pv.py                       PV名パーサ（VA_FLS形式。リング判定は無い）
│   ├── flow_fetch.py                    kblogrd取得（ログ群 VA/VAFlow、実機確認済み。既定間隔30s）
│   ├── flow_judge.py                    判定コア（直近窓だけの絶対閾値判定。層: frozen/stuck_low/
│   │                                     excess_noise/glitch。CCG式ローリング基準・固定モデル学習は不要）
│   └── flow_headless.py                 定期実行（cron `--once` or 常駐ループ。全PV判定して
│                                         flow_dashboard_state.json を書く）
├── flow_detector/flow_dashboard_state.json    flow_headless.py（detector_headless.py 相乗り／単独実行）が
│                                     書く。dashboard.py の「流量計異常検知」タブが読む（実行時生成）※拡張
├── dashboard_state.json             ← state_builder が生成
├── README.md
└── legacy/                          末次プログラム本体＋実行時データ
    ├── Anomaly_Detection_112p.py            （末次さんのオリジナル。ファイル名・中身とも無改変）
    ├── model_*.h5 (6)・sms_*.txt (6)・ALL_SDM_* (6) … モデル/標準化/SGM重み
    ├── *.sh (25)・No_*.png (3)
    └── *_FNN_*・*_Manual_*・*_Date_Range_*・*_Abort_Time_List*・*Time_Data_Abort_* … 蓄積/状態
```

- **detector_headless.py**：`legacy/` の本体を一行も書き換えずに import し、GUI を出さずに判定だけ回す。
  CCGの検知サイクルに、IP judge・温度計judge・機器劣化judge・流量計judgeを段階分散
  （`STAGE_STAGGER_SEC`、既定5分間隔）させて相乗り実行する（機器劣化judgeのみ既定で1日おき、
  流量計judgeはCCG/IP/温度計と同じ4hおき、詳細は第II部 7 章）。
- **state_builder.py**：検知が書いた蓄積ファイル（`legacy/`）を読んで `dashboard_state.json` を作る（読み取り専用）。
- **dashboard.py**：その JSON（＋ `temp_detector/temp_dashboard_state.json`、`temp_detector/temp_equipment_state.json`、
  `flow_detector/flow_dashboard_state.json`）を読んでブラウザに表示する（ポート 18050、`--port`/`DASHBOARD_PORT` で変更可）。
- **ccg_pv.py**：監視 CCG リストを CSV から読む。
- **cause_infer.py**：推定原因を keras なしで再現する（第II部 9 章）。
- **sysload.py**：共用サーバーの負荷と、本プログラム自身の使用量を読む。
- **beamcurrent.py**：蓄積電流 PV（`BMLDCCT:CURRENT` / `BMHDCCT:CURRENT`）からリアルタイムのビーム電流を読む。
- **singleton.py**：二重起動防止の PID ロック（ポート別）。
- **temp_detector/**：温度計異常検知一式（詳細は第II部 14 章）。
- **flow_detector/**：冷却水流量計異常検知一式（詳細は第II部 15 章）。
- **LER_CCG_PV.csv / HER_CCG_PV.csv**：監視対象 CCG（1列目が PV名）。

本体は `.sh`・モデル・結果ファイルをすべて作業ディレクトリ相対で読み書きするため、`detector_headless.py` は
判定前に `legacy/` へ移動して動かします。CSV と `dashboard_state.json` はトップに置いたまま参照できるよう、
各スクリプトが自分の場所を基準に絶対パスで解決します。

---

## 5. 初回セットアップ（python を 3.9.17 で呼べるように）

ログインシェルは tcsh。既定の `python` は古く（Python 2）、そのままだと辞書内包表記などで
`SyntaxError` になります。`python script.py` とフルパスなしで動かせるよう、**`~/bin/python` という
シンボリックリンク**を作って PATH に通します。

> なぜエイリアスではなくシンボリックリンクか：tcsh の **エイリアスは `nohup`・バックグラウンド実行・
> スクリプト経由では効かない**ため、以前はフルパスが必要でした。一方 **PATH は環境変数なので子プロセス
> （nohup 等）にも継承されます**。`~/bin` を PATH に通しておけば、`python script.py` が `nohup` 含め
> どこでも 3.9.17 で動きます。

```tcsh
mkdir -p ~/bin
ln -sf /cont/python/x86_64-AlmaLinux9/3.9.17/bin/python3 ~/bin/python
# ~/bin を PATH の先頭に（既にエイリアスを入れていたら削除してよい）
echo 'set path = ($HOME/bin $path)' >> ~/.cshrc
source ~/.cshrc
which python          # → ~/bin/python ならOK
python --version      # → Python 3.9.17
```

これ以降、本マニュアルのコマンドは（`nohup` のものも含め）すべて `python ...` でフルパス不要で動きます。

推定原因の表示には **h5py** が必要です（無くてもダッシュボードは動き、原因欄が「—」になるだけ）。

```tcsh
python -m pip install h5py --user
```

> それでも明示的にフルパスで呼びたい場合は
> `/cont/python/x86_64-AlmaLinux9/3.9.17/bin/python3 script.py` でも動きます。

### 動作確認済みバージョン（重要：Keras は 2 系を維持）

| ライブラリ | 動作確認済みバージョン |
|-----------|----------------------|
| Python | 3.9.17 |
| TensorFlow | 2.13.0 |
| Keras | 2.13.1 |

検知本体（`legacy/Anomaly_Detection_112p.py`）は学習済みモデルを **`.h5` 形式**で読み込みます
（`keras.models.load_model('model_*.h5')`）。

**Keras 3（TensorFlow 2.16 以降に同梱）では、この `.h5` モデルの読み込みが壊れます。**
そのため **Keras 2 系を維持してください**。`pip install --upgrade tensorflow` 等で不用意に上げると、
`load_model` が失敗して検知が動かなくなります（過去に発生した「動かなくなった」事象はこれが原因と考えられます）。

- 現状を確認：`python -c "import keras, tensorflow; print(keras.__version__, tensorflow.__version__)"`
- どうしても Keras 3 に上げる必要が生じた場合は、モデルを `.keras` 形式へ保存し直し、`legacy` の
  読み込み行（`model_*.h5` → `model_*.keras`）を差し替える移行が必要です（コード内にその下書きが
  コメントで残っています）。これは検知本体に手を入れる作業なので、慎重に。
- なお、ダッシュボードの推定原因表示（`cause_infer.py`）は keras を使わず h5py で `.h5` を直接読むため、
  この問題の影響を受けません（第II部 9 章）。

> `/cont/python` が共用環境の場合、バージョン変更は他ユーザーに影響します。共用なら**現状維持**が安全です。

---

## 6. ダッシュボードの起動（サーバー側）

```tcsh
cd ~/skbva_anomaly_detector
python state_builder.py     # 蓄積ファイル → dashboard_state.json を生成
python dashboard.py         # JSON を読んでブラウザ配信（http://localhost:18050）
```

`dashboard.py` を起動したターミナルは**開いたまま**にします（閉じるとサーバーも止まります）。

ログアウトしても動かし続けたい（共用ダッシュボードとして常駐させたい）場合は、`nohup` でバックグラウンド実行します。

```tcsh
nohup python dashboard.py >& dashboard.log &
```

- これで問題ありません。ダッシュボードは単純な HTTP サーバーなので `nohup` 常駐に向いています。
  ログアウト後も動き続け、各メンバーは各自トンネルを張って見られます。
- 出力は `dashboard.log` に出ます。`>&` は tcsh の書き方（`2>&1` は使えません）。
- 起動・配信は `kekb-co-user01` 上で行うこと（メンバーのトンネル先と同じホスト）。
- 二重起動はできません（`.dashboard.<port>.lock` とポート占有で防止。ロックはポートごとに分かれるので、
  本番を止めずに別ポートでデモ起動することもできる＝下記参照）。すでに同じポートで起動中なら
  「既に PID xxxx で起動中」と出て終了します。
- 止めるときは `pkill -f dashboard.py`（または該当 PID を kill）。

メンバーは第I部のとおり各自トンネルを張って同じ画面を見ます（サーバーは1つでよい）。

### 手元で見た目だけ確認する（実機・kblogrd 不要）

検知を回さずに、ダッシュボードの見た目（サイドバーのトレンド、PV選択時の生データプロット）を
Mac 等で確認したいとき。環境変数 `RECORD_RAW_DEMO=1` を立てて起動すると、state も詳細ビューの
生データも合成データになり（`dashboard_state.json` があっても無視される）、**5タブすべて**
（圧力異常検知・イオンポンプ異常放電・温度計異常検知・機器劣化検知・流量計異常検知）が
一貫してダミーデータで表示される。

```bash
cd skbva_anomaly_detector
RECORD_RAW_DEMO=1 python3 dashboard.py              # http://localhost:18050
# 確認したら Ctrl-C
```

**本番を止めずに別ポートで確認したい場合**は `--port`（または `DASHBOARD_PORT` 環境変数）を付ける。
ロックファイルがポートごとに分かれるため、本番（18050）と共存できる。

```bash
RECORD_RAW_DEMO=1 python3 dashboard.py --port 18077        # 本番(18050)とは別に起動できる
```

- サイドバー右のトレンド（縦軸 Anomaly Count・縦書きラベル・Period 目盛り）が表示される
  （デモ state のダミー series には `abnormal` が入っているため）。
- PV カードをクリックすると、合成の「圧力 vs ビーム電流（散布図＋2入力モデル回帰曲線）」
  「圧力＆ビーム電流 vs 時刻」が出る（圧力軸が `1.31e-6` 形式で表示される）。
- イオンポンプタブ・温度計タブ・機器劣化タブ・流量計タブにもダミーの異常データ（sev1〜3混在。
  機器劣化タブは linear/hom両モデルの表示例と未学習リングのスキップ表示例を、流量計タブは
  値の固着/校正基準比の低下/指示値不安定の3パターンを1件ずつ含む）が表示される。
- これは見た目確認専用。実データではないので数値に意味は無い。実運用では環境変数を付けないこと。

---

## 6.5 教師ラベルの保存（Normal / Abnormal ボタン）

CCG 異常カードの詳細にある **「Normal として保存」「Abnormal として保存」** は、人手で正解
ラベルを付ける機能。CCG 検知（FNN）の再学習に使う教師データを**溜める**ためのもので、
**押した瞬間に精度が上がるわけではない**（再学習は別途人手で回す）。二段構え：

1. **ボタン**：押すと `/api/label` 経由で `label_queue.jsonl` に1行追記するだけ（ring・record・
   period・abort_time・Normal/Abnormal・時刻）。**legacy の学習ファイルには触らない**ので安全。
   重い特徴量計算もしない。確認ダイアログ → 成功でトースト表示。
2. **反映（実機で人が実行）**：`apply_labels.py` がキューを読み、各ラベルについて legacy の
   解析パイプライン（`Define_Date_Range → Get_Fit_STD/CHK → Get_DIF`）でその abort timing の
   キャッシュを作り直し、legacy の `Save_Manual_*` を呼んで
   `{Ring}_Manual_{Normal|Abnormal}_Class2_Result_{Strg|Tail}_{WB|NB}.npy/.txt` に
   **byte 互換の教師行**を追記する。`Find_Abnormal_*` は呼ばないので蓄積（トレンド）は汚さない。

```bash
python apply_labels.py --dry-run   # 何を処理するか一覧（legacy は呼ばない）
python apply_labels.py             # queued を順に反映（実機・kblogrd 必須）
```

- `apply_labels.py` は legacy（`Anomaly_Detection_112p.py`）を**一行も書き換えず** import して
  関数だけ呼ぶ（`detector_headless.py` の足場を再利用）。kblogrd で当時のデータを引き直すため
  **実機（kekb-co-user01 等）で実行**すること。手元では取得段で失敗する。
- 処理後はキューの各行に `status`（applied / error:…）を書き戻す。再実行で applied はスキップ。
- 再学習そのもの（新しい `model_*.h5` を作る）は本ツールの対象外で、別途人手で行う。
- ⚠ 注意：ボタンは「今表示している異常カード」を、その `abort_time` に紐付けて保存する。
  古い fill のカードをラベルする場合でも `abort_time` で当時のデータを引き直すので整合する。

---

## 7. 検知の実行（`detector_headless.py`）

**EPICS/kblogrd のある実機でのみ動きます。** 判定後に `dashboard_state.json` を更新します。

```tcsh
cd ~/skbva_anomaly_detector
# 1回だけ（リングごとに JSON 更新。完了で「=== --once 完了 …」。IP/温度計/機器劣化/流量計judgeも1回走る）
python detector_headless.py --once

# イオンポンプ judge だけ1回（ip_judge_state.json を更新。初回投入/テスト用）
python detector_headless.py --ip-judge
# 判定窓を指定したいとき（既定は直近24h）
python detector_headless.py --ip-judge --hours 72                       # 直近72時間
python detector_headless.py --ip-judge --end 20260620000000 --hours 48  # 過去の特定期間を狙う

# 温度計 judge だけ1回（temp_detector/temp_dashboard_state.json を更新。初回投入/テスト用）
python detector_headless.py --temp-judge

# 機器劣化 judge だけ1回（temp_detector/temp_equipment_state.json を更新。learn済みモデルが
# 無いリングは自動スキップ。初回投入/テスト用）
python detector_headless.py --equipment-judge

# 流量計 judge だけ1回（flow_detector/flow_dashboard_state.json を更新。ビーム電流と無関係・
# 直近窓だけの判定なので学習等の事前準備は不要。初回投入/テスト用）
python detector_headless.py --flow-judge

# アボートトリガ常駐（推奨：アボートで即解析＋定期チェック。IP/温度計/機器劣化/流量計judgeも段階分散して相乗り）
# 端末を開いたまま、フォアグラウンドで動かす。停止は Ctrl-C。
python detector_headless.py --watch

# 単純な定期ループ（4時間間隔）
python detector_headless.py
```

完走に数分〜十数分かかります（約600本×数日分のデータ取得と曲線あてはめ）。

> **CCG・イオンポンプ・温度計(LER)・温度計(HER) は同時に kblogrd へアクセスすると負荷が集中するため、
> `STAGE_STAGGER_SEC`（`detector_headless.py` 冒頭の定数、既定 **5分**）ずつ段階的にずらして実行します**
> （CCG → 待機 → イオンポンプ → 待機 → 温度計LER → 待機 → 温度計HER）。0にすれば従来どおり即時連続実行に戻せます。

> **`--watch` は `nohup` を付けず、フォアグラウンドで動かすことを推奨します。**
> `nohup ... &` でバックグラウンドに回すと、**動いていることが画面から見えなくなり、
> 別のホストや別の端末で気づかずに二重・三重に起動してしまう**おそれがあります。
> 検知を複数同時に走らせると NFS 共有ファイルが壊れます（後述「複数ホスト同時実行による破損」）。
> 常駐させたい場合は、画面が残る `tmux`／`screen` の中でフォアグラウンド実行するのが安全です
> （`tmux` 内で `python detector_headless.py --watch` を動かし、デタッチして放置、再アタッチで状況確認）。

`nohup` での常駐も技術的には可能ですが、上記の二重起動リスクがあるため非推奨です。どうしても使う場合は、
起動前に必ず全ホストで `ps -u <user> | grep -i python` を確認し、既存の検知が無いことを確かめてください。

```tcsh
# 非推奨（二重起動に注意。tcsh なので >&。2>&1 は不可）
nohup python detector_headless.py --watch >& watch.log &
```

共用機の負荷を下げたい場合は優先度を落として実行できます（`nice` は tcsh 組み込みと衝突するので GNU nice をフルパスで）。

```tcsh
/usr/bin/nice -n 19 python detector_headless.py --watch
```

### `--watch`（アボートトリガ）の動き

末次プログラムの「Abt Trg 自動モード」のヘッドレス版です。アボート PV
（LER `CGLSAFE:MR:ABORT` / HER `CGHSAFE:MR:ABORT`）を監視し、

- アボート（値が非0）で、前回実行から30分未満なら抑制、それ以外は**3分待って**（Tail データが揃うのを待って）解析。
- アボートが無くても**最低4時間ごと**に定期チェック。
- LER/HER の解析は直列化（同時実行による競合を防止）。

### ビームあり／なしの分岐

末次プログラムの `Main_Command` に合わせて、Tail 解析の有無を切り替えています。

- **基準期間にアボートが無い**（Tail の基準が作れない）場合：Tail をスキップし、Storage のみ判定。
- **ビームはあるが今フィルに Tail データが無い**場合：同じく Storage のみ判定。
- Storage も Tail も揃っている場合：両方を判定。
- ビームが無い（停止期間）場合：no-beam 用の処理で Storage・Tail を判定。

### 表示する異常の新しさ

ダッシュボードは「いま起きている異常」を映すため、**直近の検知時刻から `RECENT_DAYS`（既定3日）以内**の
異常だけを現在の異常として表示します。停止期間など新規異常が無い間は「直近N日に異常なし（OK）」になり、
過去の履歴は「最後に記録された異常」として小さく添えるだけにします。`RECENT_DAYS` は `state_builder.py` の
定数で調整できます。

---

## 8. 監視 CCG リスト（CSV）

監視対象は `LER_CCG_PV.csv` / `HER_CCG_PV.csv` の1列目（PV名）で管理します。
`detector_headless.py` がこれを読み、本体の CCG リストを CSV ベースに差し替えます。
CCG が増減したら **CSV を更新するだけ**でよく、本体のソースを編集する必要はありません。

- ダッシュボードの「監視中の真空計」台数は、この CSV の行数から数えています（現在 LER 308 / HER 297）。
- 判定対象そのものは各側室の `.sh`（kblogrd）が取ってくる実在 CCG です。CSV と `.sh` の範囲は揃えておくこと。
- CSV には S座標も含まれるので、将来セクション配置を実配置に寄せた図を作りたくなれば利用できます
  （現行ダッシュボードのリング配置図は CCG カードと重複するため削除済み）。

---

## 9. 推定原因と圧力トレンドの仕組み

どちらも検知本体には手を入れず、すでに保存されている結果ファイルの列だけから作っています。

- **推定原因（`cause_infer.py`）**：末次プログラムの原因推定（`Find_Possible_Cause`）と同じ計算を、
  keras を使わず純 numpy で再現します。原因モデルは小さな2層ニューラルネット（`model_pc_*.h5`）で、
  保存された重みをそのまま順伝播すれば元の判定と一致します。入力・標準化・原因ラベルは本体と同一です。
  必要なのは h5py だけで、無い環境では原因欄が「—」になるだけで他は動きます。

  なぜ keras を使わないか（軽量化の狙い）:
  - **共用機にやさしい**。tensorflow/keras は import するだけで数百MBのメモリと数秒の起動を要する。
    ダッシュボードの表示（原因ラベル付け）のためだけに重量級フレームワークを常駐させずに済む。
  - **依存が少なく移植性が高い**。必要なのは numpy と h5py だけ。tensorflow はバージョンや GPU 周りで
    環境を選びやすいが、numpy+h5py ならまず問題にならない。
  - **結果は同一**。簡略化や近似ではない。モデルは `Dense(tanh)→Dense(softmax)` の2層で乱数も無く、
    `.h5` の重みをそのまま順伝播すれば keras と数学的に同じ argmax になる（softmax は単調なので
    argmax は変わらない）。「軽いが別物」ではなく「軽くて同じ答え」。

  > 念のための確認：numpy 再現が GUI 版（keras）と一致することは原理的に保証されるが、運用開始時に
  > **一度だけ**、同じ異常についてダッシュボードに出る推定原因と、レガシー GUI 版
  > （`legacy/Anomaly_Detection_112p.py`）が出す推定原因を突き合わせて、ラベルが一致することを
  > 確認しておくと安心。
- **圧力トレンド**：生の時系列は保存されないため、結果ファイルの**フィルごとの要約値**から
  「直近Nフィルの圧力・ビーム電流」を組み立てています。同じビーム電流でも圧力が上がっていく様子
  （異常の進行）が見えます。

---

## 10. トラブルシューティング（サーバー側）

| 症状 | 対処 |
|------|------|
| `nohup ... > log 2>&1` で「曖昧な出力リダイレクトです」 | tcsh では `2>&1` は使えない。`>& log` を使う。 |
| `nice: 番号が正しい書式になっていません` | tcsh 組み込みの nice。GNU nice をフルパスで `/usr/bin/nice -n 19 ...`。 |
| 実行すると Python 2 で `SyntaxError`（辞書内包表記など） | `python` が古い Python 2 を指している。第II部 5 章の `~/bin/python` シンボリックリンクを作り PATH に通す（エイリアスは nohup 等で効かない）。応急的にはフルパス `/cont/python/.../3.9.17/bin/python3 script.py`。 |
| サーバー起動時 `Address already in use` | そのホストで 18050 が使用中。`dashboard.py` の `PORT` を変え、トンネルも揃える。 |
| 「既に PID xxxx で起動中」 | 二重起動防止。動いている方を使うか、止めてから起動。残骸ロックは次回起動時に自動掃除。 |
| ポート使用確認 | `ss -ltnp | grep 18050`。 |
| 検知が `could not convert string to float: '攀'` / `chr() arg not in range` で異常0件 | 複数ホスト同時実行による共有ファイル破損。下の「複数ホスト同時実行による破損」を参照。検知は1ホストに固定する。 |
| 温度計/機器劣化/流量計タブに「⚠ アーカイバ(kblog)が…停止中」バナーが出る・バッジが「停止中」 | 故障ではない。シャットダウン期間等でアーカイバが該当PVの記録を止めている（判定に使うデータ自体が無い）状態。アーカイバ復旧後の次回判定サイクルで自動的に通常表示に戻る。過去期間で判定ロジックだけ確認したい場合は `flow_headless.py --once --end <過去日時>` などが使える（第II部 15 章）。 |
| `ip_judge.py learn` / `temp_equipment.py learn` 等で `subprocess.TimeoutExpired` の長いトレースバックが出て止まる | kblogrd がタイムアウト（既定300s）以内に応答しなかった。`learn` は数日〜数週間分をまとめて取りに行くため、通常の24h判定より起こりやすい。まず自動で本数を減らして再試行するので大抵はそのまま完了するが、それでも解消しない場合に表示される案内文の対処法（環境変数 `IP_KBLOGRD_TIMEOUT`/`TEMP_KBLOGRD_TIMEOUT`/`FLOW_KBLOGRD_TIMEOUT` でタイムアウトを延ばす、期間を区切る、負荷が下がってから再実行する）に従う。 |
| `--watch` 実行中に `top` で見ると CPU使用率が1000%超（10コア分以上）に達する | 実機で確認・修正済み。numpy(OpenBLAS)・TensorFlowは既定で「使えるだけのコア数」を毎回の計算にフル動員しようとするが、本プログラムは多数のPVを逐次ループしながら小さな配列に統計計算をかけるため逆効果でしかない上、`--watch`はCCG本体・IP・温度計・機器劣化・流量計のjudgeを別スレッドで並行実行する設計なので、各スレッドが全コアを奪い合い多重に競合していた。`detector_headless.py`冒頭でOMP_NUM_THREADS等を1に固定する対策を入れ、`ip_judge.py`/`temp_equipment.py`/`temp_headless.py`/`temp_batch.py`/`temp_judge.py`/`flow_headless.py`/`flow_judge.py`（`learn`やcronでの単独実行、selftestも含め`--watch`と同時に動きうる全エントリポイント）にも同じ対策を入れたので、最新版なら発生しない（対策前のバージョンで発生していた場合は最新版に更新すること）。この問題が起きても、以前はダッシュボードの「本プログラムCPU」がdetector_headless.py（別プロセス）を計測していなかったため気づけなかった。現在はダッシュボードに「検知プログラムCPU/Mem」の表示があるので、これで実機のCPU高騰にすぐ気づける（第I部3章参照）。なお`top`の%CPUは1コア=100%、ダッシュボードの%は全コア合計=100%という分母の違いがあるので、単純比較する際は`top`側の値をコア数で割ること（例: 16コアなら`top`の102%はダッシュボードでは約6.4%）。 |

### 複数ホスト同時実行による破損（重要・再発防止）

**症状**：検知が `could not convert string to float: '攀'`（または `'䌀'` 等の漢字）や
`chr() arg not in range(0x110000)` で例外になり、「異常0件」のまま終わる。例外の発生場所は
`legacy/Anomaly_Detection_112p.py` の `Get_Fit_STD_Strg`（基準データの float 変換）。実行のたびに
化け文字が変わるのが特徴。

**原因**：`kekb-co-user01/02/03` は**ホームを NFS 共有**しているため、`legacy/` も共有。検知は
`legacy/` に chdir して中間ファイル（`*_Data_*.txt` や `*Record_Data*.npy` 等）を読み書きする。
**複数のホスト（またはプロセス）で検知が同時に走る**と、片方が書いている最中のファイルをもう片方が
読み、書き込み途中のバイト列を拾って 1 バイトずれた文字列として読む（UTF-32 バッファのずれ）。
これが化け文字の正体（`攀`=`'e'`(0x65)、`䌀`=`'C'`(0x43) が 1 バイト上位にずれたもの）。

> 注意：`singleton.py` の二重起動防止ロックは**ホストごと**（ローカルの `/proc` を見る）。
> 別ホストで走る検知は検知できないので、NFS 共有環境では**運用で 1 ホストに限定する**必要がある。

**対処（実際に復旧した手順）**：
1. 全ホストで検知を止める（各ホストで `ps -u <user> | grep -i python` → `pkill -f detector_headless.py`）。
2. 検知を動かすホストを 1 つに決める（例：`kekb-co-user01`）。
3. 破損した実行時生成ファイルを掃除して作り直す（再生成されるのでオリジナルには影響しない）:
   ```tcsh
   cd ~/skbva_anomaly_detector/legacy
   rm *Record_Data*.npy
   cd ~/skbva_anomaly_detector
   python detector_headless.py --once >& once.log
   ```
4. `once.log` にトレースバックが出ず、`Storage 異常 N 件` 等が出れば復旧。

**診断ツール**：`check_npy.py` で、実行時生成 `.npy` に数値変換できないセル（化け）が無いか調べられる。
```tcsh
cd ~/skbva_anomaly_detector/legacy ; python ../check_npy.py
```

**再発防止**：**検知（`detector_headless.py`）は必ず 1 ホストだけで動かす。** どのホストで動かすか決めて
固定し、他ホストでは起動しない。ダッシュボードもトンネル先ホストで 1 つだけ動かせば十分。

### 検知時刻の 10 分ラグ（`CHK_LAG_MIN`）について

`detector_headless.py` の冒頭に `CHK_LAG_MIN = 10`（分）という定数があります。これは
**定期チェック（`--once`／定期ループ／`--watch` の定期セーフティネット）が検知に使う「現在時刻」を
10 分だけ過去にずらす**ためのものです。アボートで走る本来の検知（実アボート時刻を使う経路）には
適用されません。

**なぜ必要か**：レガシーは調査期間の終端を「渡した時刻 + 数分（`minit_advance`）」で決めるため、
ラグ無しだと**終端が現在時刻に張り付きます**。一方、検知は側室 D01→D12 を**順番に**データ取得する
ので、取得に時間がかかると、後の側室ほど取得時点の時刻が進み、終端付近の 30 秒刻みの点を
1〜2 個多く拾います。すると側室ごとに時刻点数（行数）が食い違い、`Get_Fit_CHK_Strg` の
`np.append(..., axis=1)` が

```
ValueError: all the input array dimensions except for the concatenation axis must match exactly,
            but along dimension 0, the array at index 0 has size 1791 and the array at index 1 has size 1789
```

で落ちます。終端を 10 分前に固定すると、全側室が「すでに確定した同一区間」を取得するため、
取得にかかる時間に関係なく行数が揃い、この問題が起きません。データの鮮度が数分落ちるだけで、
監視用途には影響しません。

**いつ顕在化したか**：通常運転では調査期間（直近フィル）は短く、取得も速いので問題になりません。
HER で長時間（十数時間）連続でビームが蓄積し、フィルの区切りが無い状況だと取得期間が長くなり、
取得に時間がかかって初めて顕在化しました。

**調整**：万一 10 分でもまだ行数がずれる場合は `CHK_LAG_MIN` を増やします（取得所要時間より十分
大きく、例：15〜20）。

> 補足（`File_Save_Para` の型）：検知結果を蓄積ファイルに保存するには、レガシーへ渡す
> `File_Save_Para` を**整数 `1`** にする必要があります（レガシー側は `if File_Save_Para == 1:` で判定）。
> 文字列 `"Save"` だと `"Save" == 1` が偽になり、検知は走って件数は返るのに**結果ファイルが
> 保存されず**、ダッシュボードに異常が出ません。`detector_headless.py` の `CONFIG` で `1` に設定済み。

### コマンド早見表（備忘録）

前提：第II部 5 章の `~/bin/python` を作ってあれば、すべて `python ...` でフルパス不要。
tcsh なのでリダイレクトは `>&`（`2>&1` は不可）。バックグラウンドは末尾 `&`。

```tcsh
# --- 初回だけ（python を 3.9.17 に） ---
mkdir -p ~/bin
ln -sf /cont/python/x86_64-AlmaLinux9/3.9.17/bin/python3 ~/bin/python
echo 'set path = ($HOME/bin $path)' >> ~/.cshrc ; source ~/.cshrc
python -m pip install h5py --user        # 推定原因表示に必要

cd ~/skbva_anomaly_detector              # 以下すべてこのディレクトリで

# --- ダッシュボード ---
python dashboard.py                      # 前景起動（ターミナルを閉じると止まる）
nohup python dashboard.py >& dashboard.log &   # 常駐（ログアウトしても継続）
python dashboard.py --port 18077         # 本番を止めずに別ポートでデモ確認（RECORD_RAW_DEMO=1と併用）
pkill -f dashboard.py                    # 止める

# --- 検知（実機のみ。EPICS/kblogrd 必要） ---
python detector_headless.py --once       # 1回だけ（CCG+IP+温度計）
python detector_headless.py --watch      # アボートトリガ常駐（推奨：前景。停止は Ctrl-C）
#   nohup ... & は二重起動に気づきにくいため非推奨。常駐は tmux/screen 内で前景実行を推奨。
#   起動前に各ホストで ps -u <user> | grep -i python して既存検知が無いことを確認。
pkill -f detector_headless.py            # 止める

# --- 表示用 JSON を作り直す（検知を回さないとき手動で） ---
python state_builder.py

# --- イオンポンプ放電電流の異常判定（運用中。kblogrd 必要。既定は固定モデル方式） ---
python ip_judge.py learn LER <健全開始> <健全終了> --interval 300 --robust --out ip_models.json  # 初回だけ（両リング）
python detector_headless.py --ip-judge             # judge を1回（ip_judge_state.json 等を更新）。--watch 中は4hごと自動
python detector_headless.py --ip-judge --hours 72  # 判定窓を変えたいとき（既定24h）
python ip_judge.py selftest                        # 合成データで層・フィットを検証（kblogrd 不要）

# --- 温度計異常検知（temp_detector/。kblogrd 必要。CCG式ローリング基準、固定モデル不要） ---
python detector_headless.py --temp-judge           # judge を1回（temp_dashboard_state.json 更新）。--watch 中は4hごと自動
cd temp_detector && python temp_batch.py run HER --rolling --hours 24   # 手元で個別に確認したいとき
cd temp_detector && python temp_headless.py --once # cron 向け単発実行（下記参照）
# crontab 例（4時間おき）:
#   0 */4 * * *  cd ~/skbva_anomaly_detector/temp_detector && python temp_headless.py --once >> temp_headless.log 2>&1
cd temp_detector && python temp_pv.py && python temp_judge.py && python temp_fetch.py selftest && python temp_batch.py selftest  # kblogrd不要の自己テスト一式

# --- 機器劣化検知（temp_equipment.py。センサ故障ではなく測定対象の熱結合の劣化。全リング対応） ---
cd temp_detector && python temp_equipment.py learn IR 20220501000000 20220622090000 --model hom  # 過去の健全期間を学習
cd temp_detector && python temp_equipment.py judge IR 20260301000000 20260401000000               # 直近と比較（--model省略時は自動判定）
python detector_headless.py --equipment-judge      # judge を1回（learn済み全リングをtemp_equipment_state.json に更新）。
                                                    # 既定では --watch/--once/定期ループにも1日おきで自動相乗り
cd temp_detector && python temp_equipment.py judge-all   # 上と同じことを detector_headless.py 無しで手元確認したいとき
cd temp_detector && python temp_equipment_plot.py IR 20260301000000 20260401000000                # 比較散布図を出力
cd temp_detector && python temp_equipment.py selftest                                              # kblogrd不要の自己テスト

# --- 流量計異常検知（flow_detector/。ビーム電流と無関係。リング概念無し・直近窓だけの絶対閾値判定） ---
python detector_headless.py --flow-judge           # judge を1回（flow_dashboard_state.json 更新）。--watch 中は4hごと自動
cd flow_detector && python flow_headless.py --once # cron 向け単発実行（下記参照）
# crontab 例（4時間おき）:
#   0 */4 * * *  cd ~/skbva_anomaly_detector/flow_detector && python flow_headless.py --once >> flow_headless.log 2>&1
cd flow_detector && python flow_pv.py && python flow_fetch.py selftest && python flow_judge.py  # kblogrd不要の自己テスト一式

# --- 確認系 ---
python -c "import keras, tensorflow; print(keras.__version__, tensorflow.__version__)"  # 2.13系か確認
ss -ltnp | grep 18050                    # ダッシュボードのポート使用確認
which python ; python --version          # 3.9.17 を指しているか
```

メンバー側（自分の PC から見るだけ）:

```sh
ssh ckdash                               # ~/.ssh/config に設定済みなら（第I部 1 章）
# ブラウザで http://localhost:18050 を開く
```

---

## 11. 元の検知プログラム（tkinter GUI 版）の起動

`Anomaly_Detection_112p.py`（末次さんのオリジナル）を GUI で動かす場合は X11 が必要です。
本体と実行時データは `legacy/` にあるので、`legacy/` に入って起動します。

```tcsh
ssh -XY your_account@kekb-login1
ssh -XY your_account@kekb-co-user01
cd ~/skbva_anomaly_detector/legacy
python Anomaly_Detection_112p.py
```

- TensorFlow の `Could not find cuda drivers` / `TF-TRT Warning` は無害な情報メッセージです。
- このプログラムは推論専用で、動かし続けても自動で学習し直すことはありません
  （再学習が必要なときは別途オフラインで `.h5` を作り直します）。
- 末次さんのオリジナル（`/nfs/sadstorage-users/suetsugu/ML/`）は触らず、自分のコピーだけ扱ってください。

---

## 12. 拡張：イオンポンプ放電電流の監視（観測パネル・現在は非表示）

> **（補足）** この章は監視（判定なしのトレンド表示）系の記述です。取得（`ip_fetch`）と PV 種別判定（`ip_pv`）は
> 実装済みですが、ここで説明する**観測パネルは現在ダッシュボードでは非表示**にしてあり、異常判定は §13
> （`ip_judge`）の専用モニターに一本化されています（`ip_state`/`ip_data.json` の経路は残置・再表示可）。

CCG（圧力）に続く監視対象として、イオンポンプの放電電流の監視を進めています。
どの機器も EPICS PV になっているため、CCG と同様に値を取得できます。

### PV リストと電源種別

- 監視 PV は `LER_IP_PV.csv` / `HER_IP_PV.csv`（1列目が PV 名）で管理します（現在 LER 308 本 / HER 293 本）。
- `ip_pv.py` がこれを読み、リング・側室（D01〜D12）・**電源種別**を付与します。
- **電源は2系統が混在**します。放電電流の振る舞いが異なるため、将来の異常判定では種別ごとに分けて扱う必要があります。

| 電源 | PV 名の特徴 | 対象セクション | 例 |
|------|-----------|--------------|-----|
| KEK 電源（標準） | `_IP_` を含む | D05・D07 以外 | `VALIP:D01_IP_L01:CUR` |
| Agilent 4UHV Controller | `_4U_` を含む | **D05・D07 のみ** | `VALIP:D05_4U_L01_A01C1:CUR` |

（Agilent 4U の PV 末尾 `A01C1` は制御器番号・チャンネル。電源種別は `ip_pv.py` が `_4U_` の有無で自動判別します。）

### 取得（履歴トレンド）— `ip_fetch.py`

放電電流の**履歴**を kblogrd で取得します（現在値1点ではなく時系列。変化＝異常の兆候を見たいため、また
将来の判定がいずれも時系列前提のため、最初から履歴で取る方針）。CCG が既に kblogrd で履歴を取っているので、
同じ仕組みをイオンポンプ電流 PV に向けるだけです。レガシーには手を入れず、独立した取得部として実装しています。

```tcsh
# 実機での使い方（kblogrd 必要）。期間は yyyymmddhhmmss
python ip_fetch.py LER 20260618000000 20260619000000
```

`fetch_history(ring, start, end, interval_sec=60)` が、PV を分割して
`kblogrd -r PV,... -t <start>-<end>d<秒> -f kaleida VA/IPump` を呼び、`{pv: {section, supply, series}}` を返します。
`series` は `(時刻, 値)` の並びで、欠測や非数値は `None` として保持します。

> kblogrd の最後の引数「ログ群」は、CCG が `VA/CCG`、ビーム電流が `BM/DCCT`、
> **イオンポンプ電流は `VA/IPump`**（実機確認済み）。`ip_fetch.py` の `LOG_GROUP` に設定済みです。

### 表示（セクション一覧＋クリックで個別PVトレンド）— `ip_state.py` ＋ ダッシュボード

取得した履歴を、ダッシュボードに載せやすい形に要約します（判定はまだ無く、トレンド表示が目的）。

- `ip_state.collect_and_save(start, end)` が、両リングを `ip_fetch` で取得し、**セクション（D01〜D12）ごとに集約**して
  `ip_data.json` に保存します。各 PV は最新値とダウンサンプルしたトレンド（既定40点）を持ちます。
  `1e-10`（NODATA）は欠測として統計から除外し、トレンドでは線を切ります。
- `state_builder.py` は `ip_data.json` があれば読み込み、`dashboard_state.json` に **`ion_pumps`** ブロックとして
  載せます（無ければ載せず、ダッシュボードでも非表示）。CCG の異常項目には `device_type: "CCG"`、
  イオンポンプ側は `device_type: "IonPump"` が付き、機器種別を区別できます。
- このパネル「**イオンポンプ 放電電流（判定なし）**」は**現在は非表示**（§13 の judge モニターに一本化。
  `dashboard.py` の `renderIonPumps` を戻せば再表示可）。表示していた内容は、リング×セクションの一覧で、
  各セクションは電源種別バッジ（KEK / 4U）・有効本数・最新の最大放電電流・相対バーを表示。セクションを
  クリックすると、その中の各 PV の最新値と放電電流トレンド（対数スケール）が開きます。

実機での収集（kblogrd 必要。検知とは独立に実行できる）:

```tcsh
# 期間を指定して収集 → ip_data.json を更新 → 次回 state_builder で反映
python ip_state.py 20260610000000 20260611000000
```

> 放電電流は桁が広い（1e-10〜1e-6 A）ので、トレンドは対数スケールで描きます。色や相対バーは
> 「放電電流の相対的な大きさ」を示すだけで、**異常かどうかの判定ではありません**（判定は今後）。

### 今後の進め方

1. ~~**取得**：CCG 同様の kblogrd で履歴を取得~~ → `ip_fetch.py` で実装済み（ログ群 `VA/IPump`、実機取得確認済み）。
2. ~~**データ構造の一般化**：`device_type` 追加~~ → CCG/IonPump を区別、`ion_pumps` ブロックを新設（実装済み）。
3. ~~**判定なしでトレンド表示**~~ → `ip_state.py` ＋ ダッシュボードのイオンポンプパネルで実装済み。
4. **異常判定（実装済み → §13）**：電源種別（KEK / Agilent_4U）ごとに分けて設計し、`ip_judge.py` として実装済み。
   ダッシュボードにも専用セクションで載る（§3・§13）。判定は「無ビーム時の放電」「自分の平常電流（学習バンド）からの
   逸脱」「同一地点 CCG 圧力との整合性（`I=a·P^b`）」を併用する。絶対しきい値だけでなく自己平常・ペア整合性を見るため、
   電源ごとの電流レベルの違いに左右されにくい。さらに急性/慢性を仕分けて表示する。
   実データでも KEK 電源（変動が大きく時々下限）と Agilent 4U（滑らかで安定）で振る舞いが異なることを確認済みで、
   判定を種別ごとに分ける裏付けになっている。

   **CCG とイオンポンプの対応（同一地点ペア）**：リング内で CCG とイオンポンプは基本 1 対 1・同じ場所に設置されている。
   対応は PV 名から機械的に導ける。

   - イオンポンプ名から `IP_`（KEK 電源）または `4U_`＋末尾 `_AnnCn`（Agilent）を除くと CCG 名になる。
     例 `VALIP:D01_IP_L08:CUR` ↔ `VALCCG:D01_L08:PRES`、`VALIP:D05_4U_L01_A01C1:CUR` ↔ `VALCCG:D05_L01:PRES`。
   - チャンネル名の**末尾英字は保持**する（`D04_IP_L09A` ↔ `VALCCG:D04_L09A:PRES`、`L16X` 等）。
   - 対応相手が無いものは、CCG かイオンポンプの一方だけが設置されている地点（例 `VALIP:D01_IP_L01:CUR` は
     在るが CCG `D01_L01` は無い）。判定ではペアが揃う地点だけ整合性を見て、片側のみの地点は個別に扱う。
   - 現データでの突き合わせ結果：LER は IP 308 本中 304 本、HER は 293 本中 291 本がペア成立（規則外 0）。

   **観察ツール `ip_observe.py`（判定前のデータ確認）**：判定を作る前に、同一地点ペアの圧力と放電電流が
   実際にどう相関するかを目で確かめる。指定期間・地点ペアについて、CCG 圧力（`ccg_fetch`）とイオンポンプ
   放電電流（`ip_fetch`）を取得し、1 ペア 1 枚の PNG（上段=時系列の2軸重ね描き／下段=圧力 vs 電流の log-log
   散布図、相関係数つき）を保存する。ヘッドレス（Agg）。KEK と Agilent 4U を混ぜて選ぶので、電源種別による
   相関の違いも見える。

   ```tcsh
   python ip_observe.py LER 20260610000000 20260611000000            # KEK/4U混在で6ペア
   python ip_observe.py LER 20260610000000 20260611000000 --section D05   # D05(4U)だけ
   python ip_observe.py LER 20260610000000 20260611000000 --pairs 8 --out observe
   ```

   PNG は `observe/` に保存される。手元に持ってきて（rsync 等）、(1) 正常時に圧力と放電電流がきれいに相関するか、
   (2) その相関が KEK と Agilent 4U で違うか、を確認する。これが判定方針（ペアの整合性を見る）の成否を分ける。
     片側のみは各リング数本程度。

定期的に収集したい場合は、`python ip_state.py <開始> <終了>` を cron 等で回すか、検知サイクルに合わせて
実行します（検知本体とは独立。共用機の負荷を避けるため、数時間ごと程度で十分）。

---

## 13. 拡張：イオンポンプ放電電流の異常判定（`ip_judge` / 運用中）

セクション 12 の「監視（判定なしトレンド表示）」の次の段階として、放電電流の**異常判定**を実装し、
`detector_headless` の検知サイクルに相乗りで自動実行・ダッシュボード表示まで組み込み済みです（CCG
パイプラインより新しい機能です）。主目的は **HV フィードスルーでの放電／破損の事前検知**（大規模リークに
発展しうるため）。

### 破損の signature（実データで確認）
破損ポンプは大きく2タイプ。**(1) 持続型**（電流が高値に張り付き、圧力・ビームから切り離される＝
デカップリング）と、**(2) 過渡スパイク型**（短時間スパイク→HV オフ。中央値には現れにくい）。両方を捕捉する。
- D12_IP_L23（KEK・持続）: 低圧で電流が健全時の ~1e-6 A から ~1e-4 A へ約 2 桁上昇。I-P 散布図が水平に張り付く。
- D07_4U_H06（4U・持続）: 電流が ~1e-3 A 帯（健全 4U は 10nA〜µA）。数ヶ月かけて階段状に上昇。
- D01_IP_H14（KEK・過渡）: 2021-07 にスパイク→HV オフ。中央値は平常並みでも、ハードシーリング
  超過点が多数 → L0a の過渡スパイク検出（超過点の数）で捕捉。D11_IP_H04（2025-05）も同型。

### 判定の層（L0a / L0b / L1 / L2、＋ビーム軸）
これらは**独立した判定ルール**に番号を振ったもので、ニューラルネットの層ではない（値を伝播せず、
最後に OR で束ねる。§17 用語集参照）。
| 層 | 内容 | モデル要否 | 主対象 |
|----|------|-----------|--------|
| **L0a 無ビーム放電** | ビーム電流が低い/無い期間にガス負荷が無いのに電流が高い → 放電。CCG にも I-P にも依存せず最も頑健 | 不要 | 両電源 |
| **L0b 正常バンド超過** | 学習した正常電流（log p95）から判定窓の中央電流が何 dex 超えるか。ビーム連続時の 2 桁ジャンプを捕捉。ポンプごとに自動較正 | 学習バンド（無ければ電源固有定数にフォールバック） | 両電源（特に KEK） |
| **L1 I-P 整合（z）** | `pred=max(a·P^b, I_floor)` をクランプし `z=log10(I/pred)/σ`。上振れ=放電、下振れ=排気劣化 | I-P モデル要 | 主に 4U |
| **L2 デカップリング** | 窓内 `r(logI,logP)`・`r(logI,log I_beam)` の崩壊。**「以前は相関していた」ものの崩壊のみ**主張（健全 KEK は元々無相関なので誤検知しない） | trust モデル要 | 主に 4U＋ビーム軸 |

ビーム電流の履歴は `beam_fetch.py`（ログ群 `BM/DCCT`、`BMLDCCT/BMHDCCT:CURRENT`、mA）で CCG/IP と同じ
時刻グリッドで取得します（リアルタイム現在値の `beamcurrent.py` とは別。こちらは履歴専用）。

### 設計上の確定事項
- **フロアの使い分け**: 上振れ(放電)=全圧力域・予測フロアクランプ／下振れ(劣化)=フロア以上・trust モデル時のみ／
  学習=フロア以上のきれいな点。
- **ロバスト学習**: `I=a·P^b` を Theil–Sen（傾き中央値）＋切片中央値で推定、σ=残差 MAD×1.4826。**numpy のみ**
  （実機に scipy が無くても動く）。`b` は `[0.5,1.5]` でクランプ、外れ/レンジ不足なら `b=1` フォールバック。
- **電流バンド**は I-P 相関の有無に関わらず常に学習（KEK のように相関しないポンプでも絶対水準は使える）。
- **学習窓の汚染対策**: ① I-P モデルが low-trust の窓は前の良モデルを保持（放電を正常学習しない）。
  ② 電流バンドの p95 が前回から大きく跳ねたら、その跳ね自体が異常 → 前のバンドを保持。
  なお初回学習が汚染窓だと正常化してしまうので、**学習は既知の正常期間を選ぶ**こと（運用）。

### severity の集約（実データ較正済み）
- **L0a 無ビーム放電** はポンプ相対化: 無ビーム電流が「学習 p95 +0.7 dex」を超える、または絶対ハード
  シーリング（既定 1e-5 A）を超えるときのみ発火（平常 moderate 電流の誤検知を排除しつつ、自己平常
  からの急上昇 D12_L23/D01_H14 と、学習窓で既に高かった個体 D01_L02 の両方を捕捉）。
  さらに**過渡スパイク検出**: ハードシーリング超過点が l0_spike_min_pts(既定5)以上あれば、たとえ
  中央値が平常でも発火（D01_H14 のような「スパイク→HVオフ」は中央値に薄められるため、超過点の
  「数」で捕捉する）。
- **decoupled_pi は 4U のみ**（KEK の I-P 相関は偶発的で誤検知源のため無効化）。
  **decoupled_beam** はビーム動的範囲が十分な窓のみ。
- **sev3（フィードスルー放電疑い・最重要）**: L0a 発火 / (上振れ＋デカップリング) / L0b 大幅超過。
- **sev2**: 電流の上振れ（L1.over / L0b.over）。
- **sev1（要観察）**: デカップリング単独 / 排気劣化（下振れ）。弱い証拠は格下げ。
- **sev0**: 正常。
- **ビーム軸 格下げ（誤検知抑制）**: sev≥2 でも「無ビーム挙動」から **ビーム由来＝正常寄り** と切り分け
  られるものは sev1 へ格下げする（＝表示ゲートで出なくなる）。相関の正負では正常/故障を切れないため、
  直接の挙動で判定する: `drop_dex = log10(med_on) − log10(med_nb)`（ビームを落として電流が下がる量）と
  `nb_excess_dex = log10(med_nb) − nb_log_p95`（無ビーム電流の平常超過量）。`drop_dex ≥ 0.30 かつ
  nb_excess_dex ≤ 0 かつ r_beam ≥ 0.50 かつ med_on/med_nb が abs_hard 未満` のとき発火。放電破損（短絡等）は
  無ビームでも電流が高い＝`nb_excess > 0` なので、必須条件 `nb_excess ≤ 0` で必ず保護され格下げされない
  （例: D03_IP_L09 は無ビームで平常を +0.8 dex 超過のため残す／D10_IP_L09 は無ビームで下がりビーム追従の
  ため格下げ）。CONFIG: `beam_downgrade_enable` / `beam_drop_min_dex` / `beam_nb_excess_max_dex` /
  `beam_r_min` / `beam_downgrade_to`。調査用に `tools/ip_beam_survey.py`（drop/nb_excess/r_beam を一覧）。

### 使い方
```bash
# 学習（既知の正常期間で。粗い間隔で十分）
~/bin/python ip_judge.py learn  LER 20260601000000 20260615000000 --interval 300 --out ip_models.json
# 判定（直近窓を評価。--out-json で結果 JSON を保存）
~/bin/python ip_judge.py judge  LER 20260615000000 20260615060000 --interval 60 --models ip_models.json
# 合成データで層・フィットを検証（kblogrd 不要・実機でも実行可）
~/bin/python ip_judge.py selftest
```
学習結果は `ip_models.json`（リング→PV→{a,b,σ,r,電流バンド,…}）に保存。判定はこれを読みます。

### 実装済み・未実装
- 済: L0a/L0b の実データ較正（ポンプ相対化＋絶対ハードシーリング＋過渡スパイク検出）、
  decoupled の誤検知抑制、acute/chronic ラベル（`kind`/`deviation_dex`）。
- 済: **ビーム軸格下げ**（無ビーム挙動 drop_dex/nb_excess_dex/r_beam でビーム由来 sev3 を sev1 に格下げ。
  既知故障 D03_IP_L09 等は保護）。
- 済: `dashboard` への judge 結果の載せ込み（**イオンポンプ専用タブ**、sev1/sev2/sev3 すべて表示＋
  acute/chronic バッジ＋「N 回連続」バッジ＋逸脱量、急性を上にソート、**右側にカウント推移プロット**
  （Anomaly Count / Judge Cycle）、カードクリックで**電流 vs 時刻**（無ビーム網掛け・学習バンド）と
  **I-P 散布図**（縦軸放電電流 [A]・`I=a·P^b` 回帰を実測点と同じ赤で重ね描き）を表示）。
- 済: **ローリング基準（オプション、既定オフ）**。`detector_headless.py` の `IP_JUDGE_ROLLING=True` に
  すると、CCG/温度計と同じく judge のたびに直近数日（既定8〜5日前）から健全モデルを作り直す
  （`ip_models_rolling.json`）。ただし実機確認の結果、放電フィット `I=a·P^b` は3日程度の基準窓では
  圧力レンジ変動が不足し、本来検知できるはずの放電を見逃す事例が出たため、**既定は固定モデル方式
  （`False`）**。固定モデルより最新の状況を反映したい場合のみ有効化を検討する。
- 未: `cause_infer.py` への I・P・ビーム 3 点測量の受け渡し（放電／排気劣化／CCG 故障／正常高負荷の切り分け）。

#### judge 結果をダッシュボードに出す手順（自動）

イオンポンプ judge は **`detector_headless` の検知サイクルに相乗り**して自動実行される。
`--watch` / 定期ループの中で `IP_JUDGE_EVERY_SEC`（既定 **4 時間**＝CCG の定期セーフティネットと同周期）
ごとに直近 `IP_JUDGE_WINDOW_H`（既定 24 時間）を両リング judge し、結果を `ip_judge_state.json`
（リング結果を並べた list `[{LER}, {HER}]`）に書き、`dashboard_state.json` を再生成する。
**イオンポンプ judge はアボートには連動しない**（CCG はアボート即解析＋4h 定期だが、放電・破損は持続性が
高く即時性が不要なため、IP は純粋に 4h 周期で回す）。
`state_builder.py` がそれを `ion_pump_anomalies` / `ip_sections` として取り込み、
ダッシュボードのイオンポンプ異常モニターに反映する。

**表示ゲート（sev1/2はそのまま・sev3は継続したものだけ）**：judge は毎サイクル多数のポンプを
sev1〜3 で拾う。ダッシュボードのカードには **sev0（正常）以外は基本すべて出す**が、**sev3 だけは
`IP_MIN_COUNT` サイクル以上続いたものに絞る**（単発の判定ノイズで sev3 バッジが暴れないようにする
ため。CCG 同様の考え方）。仕組みは累積カウント：あるポンプが sev3 のサイクルで +1、そうでなければ
−1（下限0）。カウントは `ip_judge_counts.json` に保存され、各カードに「N 回連続」バッジで表示される。
閾値は `state_builder.py` の `IP_MIN_SEV`(=3) / `IP_MIN_COUNT`(=2)。すぐ sev3 を確認したいときは
`--ip-judge` を2回叩けばカウントが 2 に達して出る（同じ24h窓を2回判定）。ビーム由来と判定された
sev3 は**ビーム軸で sev1 へ格下げ**され（上記）、sev1側の表示（そのまま出る）に回る。

**判定結果の保存先**：`ip_judge_state.json`（各ポンプの severity / kind / deviation_dex /
reason / count、ビーム軸の `beam_driven` / `severity_raw` / `metrics` を含む全結果）、
`ip_judge_counts.json`（sev3 累積カウント）、`ip_judge_history.json`（PVごとの sev3 連続カウント履歴＝
カードの推移プロット用）。いずれもパッケージ直下に毎サイクル上書き保存される。`dashboard_state.json`
にも `ion_pump_anomalies` として反映。

**前提：モデル `ip_models.json` を一度だけ作る**（judge には平常モデルが要る。毎サイクル学習
し直さない＝CCG の .h5 と同じ思想）。健全期間で learn しておく。**`--robust` 推奨**（学習窓に
紛れた異常エポックを中央値+MAD の sigma-clip で自動除去。健全データは不変）：

```tcsh
python ip_judge.py learn LER <健全開始> <健全終了> --interval 300 --robust --out ip_models.json
python ip_judge.py learn HER <健全開始> <健全終了> --interval 300 --robust --out ip_models.json
```

学習バンドは**ビーム有り/無しで分けて**保持する：`cur_log_p50/p95`＝運転中(ビーム有り)の平常
電流（L0b が使う）、`nb_log_p50/p95`＝無ビームの平常電流（L0a の相対閾値が使う）。これにより
判定時もビーム状況に応じた基準と比較でき、感度が上がる。`--robust N` で反復回数を指定可（既定2）。

**複数期間を合算して1つのモデルにする**こともできる（例：無ビーム期間で `nb` バンド、ビームあり
期間で `cur` バンドを学ぶ）。既定窓に加え `--window <開始> <終了>` を繰り返し指定する：

```tcsh
python ip_judge.py learn HER 20260125000000 20260126140000 --interval 300 --robust \
       --window 20260202000000 20260202235900 --out ip_models.json
```

> 注：同じリングを2回 learn すると**上書き**（合算されない）。複数期間を混ぜたいときは上記の
> `--window` で1コマンドにまとめること。ビーム電流は点ごとに自動で有/無を判定するのでフラグ不要。

健全期間の選び方：**判定窓の直前で、運転条件が今と近い、ビーム電流が十分振れている数日〜2週間**を、
判定窓と重ねずに選ぶ。多少の異常混入は p95＋`--robust`＋絶対ハードシーリングが吸収する。

> `learn` は長い期間・多PVを一度に取りに行くため、kblogrd が既定タイムアウト（300s）以内に
> 応答しないことがある。まず本数を自動的に減らして再試行するので大抵はそのまま完了するが、
> それでも解消しない場合は環境変数 `IP_KBLOGRD_TIMEOUT` で延ばせる（例:
> `env IP_KBLOGRD_TIMEOUT=900 python ip_judge.py learn ...`）。詳細は第10章のトラブル
> シューティング表を参照。

- モデルが無い間は judge をスキップし、その旨を1回だけ表示する（古い観測パネルのみ表示）。
- すぐ反映したいときは `python detector_headless.py --ip-judge`（judge を即1回実行）。
- 設定は `detector_headless.py` 冒頭の `IP_JUDGE_*`（間隔・窓・取得間隔・有効/無効）で調整。
- 詳細プロットの生データは `/api/ip_raw`（→ `record_raw.build_ip_view`）が直近 24h の電流・
  圧力・ビームを引いて描く。学習バンドは `ip_models.json` から読む。
- 手元で見た目だけ確認するなら `RECORD_RAW_DEMO=1`（§6）で合成データ表示。

> 注：別系統の「イオンポンプ 放電電流 [判定なし]」観測パネル（`ip_state`/`ip_data.json`）は
> judge セクションに一本化したため**非表示**にした（`dashboard.py` の `renderIonPumps` を no-op 化）。
> `ip_state` 自体は残してあるので、再表示したければ `renderIonPumps` を元に戻せばよい。

---

## 14. 拡張：温度計異常検知（`temp_detector/` / 運用中）

ビームパイプ本体の温度計（LER 1550本／HER 1260本、`VA{L,H}TMP:...` 形式）に加え、IR（衝突点周辺）
のフォーカス磁石温度計（12本、`FB_MOVE:...` 形式）のセンサ自己故障（断線・短絡・接触不良等）を検知する。
**CCG/IP とは独立した自己完結パッケージ**（`temp_detector/` 配下、CCG/IPのコードに依存しない）。

### PV 形式（2種類）

- **(A) ビームパイプ本体**（ログ群 `VA/VATemp`）：`VA{L,H}TMP:{センサID}:{位置タグ}:{付帯}`
  （例 `VAHTMP:D10_139:QD3E_11:BL`）。ring は接頭辞（VAL=LER/VAH=HER）から一意に決まる。
- **(B) IR センサ**（ログ群 `BM/BMOthers`）：`FB_MOVE:{D01|D02}:{QC1H|QC1L}[連番][:BWS[2]]:TEMP`
  （例 `FB_MOVE:D01:QC1L:BWS:TEMP`）。ビームリングは tag の H/L（H=HER, L=LER）から自動推定する。

**リング所属の例外**（配線間違い・一時的な物理移設等）は `pv_info/TEMP_RING_OVERRIDE.csv` で
上書きできる（コード変更不要、2列CSV `PV,ring` を編集するだけ）：

```csv
PV,ring
FB_MOVE:D02:QC1H:BWS:TEMP,LER          ← 配線間違い（実際はLER設置）
VAHTMP:D10_139:QD3E_11:BL,LER          ← 例: HER温度計を一時的にLERへ物理移設
```
PV名（アーカイブ上の取得先）自体は変わらないため、取得は常に元のリング側 `<RING>_TEMP_PV.csv`
から行われる（`natural_ring`。上書きは判定・ビーム相関に使う実効 `ring` のみに効く）。

### 検知の層

| 層 | 検知するもの | 概要 |
|---|---|---|
| H0 | 断線（非現実高温）/ 低温 | 絶対レンジ（既定 -40〜400℃）を外れたら sev3、レンジ接近は sev1 |
| H1 | 張り付き | 窓内で値が変化しない（分解能未満）→ sev2 |
| S | 短絡疑い | 窓内でサブ常温（≤10℃）が持続 → sev3。ビーム非依存・モデル不要（実機に常温以下が平常のセンサは無いと確認済み） |
| O | 無ビーム高温（near-open/高抵抗） | 無ビーム点の温度が一定割合アンビエント超過で持続 → sev3。ビーム発熱による正常高温とは無ビーム条件で区別 |
| N | ノイズ増大 | 差分の分散が学習ノイズ帯を超過 → sev2 |
| G | グリッチ（反転スパイク） | 1点だけ跳ねて戻る反転（前後とも急変・符号が逆）→ sev2。なめらかなビーム連動スイングは含めない |
| B | ビーム反相関 | 温度-ビーム相関が強い負（無ビームで温度上昇）→ sev2。非物理なパターン |
| I | 間欠逸脱 | 稀だが繰り返す極端値（持続ではない）→ sev2 |
| P | 上下ペア乖離 | 対センサとのΔTが学習値から乖離（一部センサのみ存在） |

### 基準の方式：CCG式ローリング（IPとは異なる）

温度計は**中央値＋ノイズ帯**という単純な統計量のみを使うため、短い基準窓（既定8〜5日前の3日間）
でも安定して学習できる。よって**CCGと同じローリング基準**（判定のたびに基準窓を学習し直す）が既定。
固定モデルファイルの保存・陳腐化管理は不要（IPの回帰フィット `I=a·P^b` とは異なり、圧力/電流レンジの
変動を必要としないため）。

### 使い方

```bash
cd temp_detector
# 全台バッチ判定（手元で個別に確認したいとき）
python temp_batch.py run HER --hours 24 --rolling --top 40
python temp_batch.py run HER --start 20260625000000 --end 20260627000000  # 期間を直接指定
python temp_batch.py list-low HER            # 学習中央値が低いセンサ一覧（常温以下が平常か確認）

# 定期実行（cron 推奨、または常駐ループ）
python temp_headless.py --once               # cron 向け単発実行
python temp_headless.py --interval-hours 4    # 常駐ループ

# 特定センサの波形・判定を見る較正/診断ツール
python temp_probe.py HER D01M095 20260615000000 20260618000000 600

# selftest（kblogrd 不要）
python temp_pv.py && python temp_judge.py && python temp_fetch.py selftest && python temp_batch.py selftest
```

`detector_headless.py` の検知サイクルにも相乗りしており（`--watch`/`--once`/定期ループいずれも）、
`--temp-judge` で単体実行もできる（第II部 7 章）。結果は `temp_detector/temp_dashboard_state.json`
に書かれ、ダッシュボードの「温度計異常検知」タブがこれを読む。

**アーカイバ停止の検知**：`temp_headless.py` は、PVリストは読めているのに判定できたPVが
0本（`stats.n_judged == 0`）の場合、個々のセンサ故障ではなく**アーカイバ(kblog)自体がデータ
取得を停止している**と判断し、その旨を `archiver_stopped: true` としてリングごとにJSONへ記録する
（シャットダウン期間など、アーカイバが動いていない間の運用を想定）。ダッシュボードはこのフラグを
見て、通常の異常表とは別に警告バナーを表示する（「異常なし」と誤解されないようにするため）。

**クリック展開プロット**：異常（sev≥1、保存対象=上位top件）のPVには、判定窓の間引き時系列
（`plot: {t, temp, beam}`、400点以下）がJSONに埋め込まれ、ダッシュボードで行をクリックすると
温度＆ビーム電流の時系列グラフが表示される。judge時点でメモリ上にあるデータをそのまま使う
ため、**クリック時のkblogrd再取得は発生しない**（正常PVには付けないのでJSONサイズも異常件数に
比例した分だけで済む）。欠測（NaN）はnullとして埋め込まれ、グラフでは線が途切れて描かれる。

ここまでの層（H0〜P）は「**センサ自体**の故障」を見る。`temp_equipment.py` は逆に、**センサは
正常な前提で、測定対象の機器側の熱結合の劣化**（放熱不良・断熱劣化・接触不良による発熱増加等）
を検知する、別の判定軸のツール。**IRセンサに限らず、LER/HER 本体センサ（`VA{L,H}TMP` 形式）も
含め、全リング・全PVで同じように使える**（PV形式ごとの分岐は内部で吸収されるため、呼び出し側は
ring を `LER`/`HER`/`IR` のどれにするか選ぶだけでよい）。

**着眼点**：健全な機器なら、同じビーム電流に対する温度上昇（dT/dI）はほぼ一定のはず。これが
過去の基準期間と比べて有意に増えていたら、機器側の熱的な劣化を疑う。実例（IR:
`FB_MOVE:D01:QC1L:BWS:TEMP`）では、2022年基準期間 dT/dI≈4.7℃/A に対し 2026年は≈7.9℃/A（比
1.68倍）で検出。

**2つのフィットモデル**（`--model` で選択）：

| モデル | 式 | 備考 |
|---|---|---|
| `linear` | `T = a + b·I` | Theil-Sen 頑健回帰。傾き `b`(dT/dI) の比で判定。Nb取得不要の簡易版 |
| `hom`（既定） | `T = w0 + w1·I + w2·(I²/Nb)²` | Suetsugu et al., PRAB **27**, 063201 (2024) 式(5)と同形（CCG圧力の式をそのまま温度に適用）。バンチ数 `Nb` の取得が必要（`temp_fetch.fetch_nb`、ログ群 `Misc/Base`、PV `CGLINJ:BKSEL:NOB_SET`/`CGHINJ:BKSEL:NOB_SET`。CSV読み込み時は列名に `NOB`/`BKSEL`/`BUNCH` のいずれかを含む列を自動検出） |

`hom` の「二乗」は本来、圧力側の熱脱離物理（Arrhenius近似 `ΔPt∝(ΔT)²`）に由来するもので、温度
そのものへの適用は必ずしも自明ではない（二乗なし版 `fit_t_vs_i_hom_linear` も用意。実データで
当てはまり(R²)を比較して判断するとよい）。実際に検証したところ両モデルとも R²=0.93〜0.99 と
良好で、`hom` の方が実データの非線形な立ち上がりをよく捉える傾向があった。

**フィットに使う点の絞り込み**（2段階、両方とも既定で有効）：
1. **ビームあり点のみ**（`beam_on_ma`、既定50mA以上）。CCGの Storage 解析（フィル中のみを見る）
  と同じ考え方。
2. **電流急変直後の熱の過渡（サーマルラグ）を除外**（`settle_after_change_min`＝直近何分を見るか、
  既定20分／`settle_change_ma`＝その間の変動幅がこれ以上なら過渡とみなす、既定200mA）。アボート
  による急落・フィル開始や電流アップによる急上昇のどちらの方向でも、電流が変わってから機器の
  温度が実際に落ち着くまでには熱容量による遅れがあるため、その間の点は「今の電流に対する定常的
  な温度」ではない。CCG論文の Storage/Tail の区分と同じ発想を、方向を問わず一般化したもの。
  **`settle_after_change_min` は機器ごとの実際の熱時定数に合わせて調整が必要**（既定20分で
  不十分なら、プロットを見ながら30分・45分などに伸ばす）。

**learn / judge の2段構成**（IPと同じ思想。機器の熱結合特性は年単位でしか動かない量なので、
CCG/温度計センサ判定のような「毎回直近数日を学習し直す」ローリング基準は使わず、明示的に選んだ
過去の健全期間を一度学習してモデルを保存する）：

```bash
cd temp_detector

# ① 過去の健全期間を学習（--model 省略時は既定で hom＝式(5)型。--model linear で簡易版に切替可）
python temp_equipment.py learn IR 20220501000000 20220622090000 --model hom

# ② 直近を学習済みモデルと比較（--model 省略時は保存済みモデルの種別を自動判定）
python temp_equipment.py judge IR 20260301000000 20260401000000

# LER/HER 本体センサも同様（PV数が多い＝1550/1260本なので、まず --pv で1本に絞って試すのが安全）。
# ビーム/Nbはリング共通なので、PV本数によらず最大2回（LER/HER分）しか取得し直さない。
# 温度自体も26本ずつまとめて1回のkblogrd呼び出しで取得する（temp_fetch.fetch_historyのCHUNK
# 機能をそのまま使う）ので、kblogrd呼び出し回数は概ね「PV本数÷26」で済む（1本ずつ呼ぶより
# 大幅に少ない。呼び出し1回あたりの接続オーバーヘッドが効くため、本数が多いほど効果が大きい）。
# 既定値26は、元のCCG用.sh（legacy/HERD01CCG.sh等）がD01のCCG27本を14本+13本に手分けして
# いたのを踏襲した「13」から、実機で26本一括取得も問題なく動作したことを確認して引き上げた値
# （kblogrd自体に本数の上限が確認されているわけではない）。さらに増やせるか試したい場合は
# 環境変数 TEMP_KBLOGRD_CHUNK で上書きできる（コード変更不要）。段階的に試すのを推奨
# （kblogrd側の未知の上限に備え、いきなり大きくしない）:
#   実機は tcsh なので "VAR=val command"（bash風）は使えない。env 経由か setenv を使う:
#   env TEMP_KBLOGRD_CHUNK=52 python temp_equipment.py learn HER 20220501000000 20220622090000 --model hom --match "D12"
#   または: setenv TEMP_KBLOGRD_CHUNK 52 ; python temp_equipment.py learn HER ...
python temp_equipment.py learn HER 20220501000000 20220622090000 --model hom --pv "VAHTMP:D10_139:QD3E_11:BL"
python temp_equipment.py judge HER 20260301000000 20260401000000 --pv "VAHTMP:D10_139:QD3E_11:BL"
# 1本で確認できたら --pv を外す（または --match でセクション等を絞る）ことでリング全体に広げられる
python temp_equipment.py judge HER 20260301000000 20260401000000

# 比較の詳細（R^2 込み）を1本だけ見る／変化ログCSVからのオフライン検証にも対応
python temp_equipment.py compare IR "FB_MOVE:D01:QC1L:BWS:TEMP" --model hom \
  --ref-csv 2022.csv --now-csv 2026.csv         # または --ref-start/--ref-end で実機取得

# モデル保存なしで1回だけざっと洗い出したいとき（learn不要の簡易版）
python temp_equipment.py scan IR 20220501000000 20220622090000 20260301000000 20260401000000

# 比較散布図（温度 vs ビーム電流、基準/現在を重ね描き。--model 省略時も自動判定）
python temp_equipment_plot.py IR 20260301000000 20260401000000

# selftest（kblogrd 不要）
python temp_equipment.py selftest
```

`judge` の出力は sev1〜3・比・代表点での予測発熱量・両期間の R² を表示する。**代表点は
両期間それぞれの実データ範囲の共通部分にクリップ**しており、データ量や電流域が違う期間同士を
比較しても外挿にならないようにしてある。

**判定は「ビーム電流ゼロの温度（環境温度）」の影響を受けない設計**：本当に検知したいのは
「同じビーム電流に対してどれだけ発熱するか」の変化であって、季節・空調等で環境温度自体が
変わることではない。`linear` モデルは切片 `a`（＝I=0での温度）を使わず傾き `b`（dT/dI）だけで、
`hom` モデルは切片 `w0` を除いた `w1·I + w2·(I²/Nb)²`（＝発熱分のみ）で比較しており、どちらも
環境温度の変化そのものは判定に影響しない（selftest で「環境温度だけが大きく変わり、真の熱結合
特性は同一」というケースが sev0 のままであることを回帰テストとして確認している）。ただし
`judge` の出力には参考情報として切片差（`linear`=Δa、`hom`=Δw0）も表示されるので、環境温度が
実際どれくらい動いたかは目視で確認できる。

**同一エリアでの一括検知への注意**：同じセクション（例 D01）の多数のPVが同時に sev3 になった
場合、`judge` が自動で警告を出す。個々の機器が独立に劣化するより、季節変化・空調・その周辺の
改修・センサ較正の変更など**共通要因**を疑うべきサインであることが多い（実データで実際に
D01エリア6本が同時検知され、運用者側で「基準期間と直近の間に温度計周りで変更があった」ことが
裏付けられた）。切片差(Δa/Δw0)も参考表示されるので、周囲温度シフトか機器結合そのものの変化かの
見立てに使える。

`temp_equipment_plot.py` は温度 vs ビーム電流の散布図を基準/現在で重ね描きし、`--model` 未指定
なら PV ごとに保存済みモデルの種別・学習期間を自動判定する。各期間のフィット曲線はその期間で
実際に観測された電流範囲までしか描かない（データの無い領域への外挿を避けるため）。図中の文字は
すべて英語（実機に日本語フォントが無い環境があるため。`ip_observe.py` と同じ方針）。

**ダッシュボード連携・`detector_headless.py` 連携**：ここまでは手元での `learn`/`judge` の
使い方だが、実運用では CCG/IP/温度計センサと同じく `detector_headless.py` の検知サイクルに
相乗りさせ、ダッシュボードの「機器劣化検知」タブ（第I部 3 章）で見る。

- `run_periodic_judge()`（`temp_equipment.py` 内）が、**learn は行わず**保存済みモデル
  （`temp_equipment_models.json`）との judge だけをリングごとに回し、`temp_equipment_state.json`
  を書く。まだ `learn` していないリングは自動的にスキップされる（エラーにはならない）ので、
  例えば「IRだけ運用中、LER/HERは未着手」という段階的な導入でも安全に動く。
- `detector_headless.py` 側は `run_equipment_judge`/`_maybe_run_equipment_judge` として
  CCG/IP/温度計と同じ相乗り方式で配線されており、`--watch`/`--once`/引数無しの定期ループの
  いずれでも自動的に実行される。ただし機器の熱結合特性は年単位でしか動かない量なので、
  既定の実行間隔は **1日ごと**（`EQUIPMENT_JUDGE_EVERY_SEC`、CCG/IP/温度計の4hよりずっと緩い）
  にしてあり、実機負荷はほぼ無視できる。
- 単体実行・初回投入・動作確認には `python detector_headless.py --equipment-judge`
  （`temp_equipment_state.json` を即時更新）が使える。`detector_headless.py` を経由せず
  手元だけで同じことを試したい場合は `cd temp_detector && python temp_equipment.py judge-all`
  でも同じ結果になる（`judge-all` は `run_periodic_judge` をそのまま呼ぶ薄いCLIラッパー）。
- `dashboard.py` は `temp_detector/temp_equipment_state.json` を読み、「機器劣化検知」タブに
  リングごとの判定結果を表示する。未学習のリングは「未学習のためスキップ」と表示されるだけで、
  他のリング・他のタブの動作には影響しない（温度計異常検知タブが未実行時に案内文だけ出すのと
  同じ設計）。learn済みリングでもjudge時点でアーカイバが停止していれば、温度計異常検知タブと
  同じ`archiver_stopped`フラグで警告バナーを表示する。未学習のリングはタブ見出しのバッジが
  「未学習」になる（学習済みで判定できている場合の件数表示・アーカイバ停止中の場合の「停止中」
  とは別の状態として区別）。また、未学習のリングについては、同じ温度アーカイバを使う
  「温度計異常検知」タブ側がその時刻に`archiver_stopped`であれば、「learn後もこの期間は同様に
  アーカイバ停止中表示になる見込みです」という参考情報を併記する（機器劣化検知自身はlearn前は
  データ取得を試みないため、未学習であることと現在のアーカイバ状態は本来別の情報だが、
  判断材料として関連付けている）。
- **クリック展開プロット**：異常（sev≥1）のPVには、`judge(attach_plot=True)` が温度 vs ビーム
  電流の散布データ（`plot: {ref, now, ref_fit, now_fit}`、各250点以下＋フィット曲線）をJSONに
  埋め込み、ダッシュボードで行をクリックすると基準期間（薄青）と調査期間（赤）の散布＋フィット
  曲線が表示される。`hom`型は判定のために基準期間の生データを毎回再取得するので基準側の散布点も
  入るが、`linear`型は保存済みフィット係数（a_ref/b_ref）しか持たないため、基準はフィット直線
  のみになる（グラフの注記にも表示される）。

---

## 15. 拡張：冷却水流量計異常検知（`flow_detector/` / 運用中）

冷却水流量計の**センサ自身の異常**を検知する（実際の流量低下は別のアラームシステムが検知する
ため対象外＝ユーザ確認済み）。運用上、これまでに実際の流量低下が起きたことは一度も無く、
流量計自身の故障で指示値が下がってしまうケースが全て（分解清掃すれば指示値が元に戻る）との
ことなので、**センサ側の異常予兆を事前に見つける**のが目的。

**ビーム電流と無関係の機器**（CCG/温度計/機器劣化検知はいずれもビーム電流との関係が判定の
軸になるが、流量計はそれが無い）なので、リング(LER/HER)の概念を持たず、判定は**直近1窓
（既定24h）の指示値だけ**を固定閾値と比較する、他の拡張よりさらに単純な設計にしている
（CCG式ローリング基準の別窓取得や、機器劣化検知のような固定モデルの学習は不要）。

### PV 形式

```
VA_FLS:{section}_{idx}_{tag}:RATE
  例) VA_FLS:D01_11_XXX:RATE, VA_FLS:D04_15_084:RATE
```
`section`=D01〜D12（CCG/温度計と同じセクション表記）、`idx`=セクション内の連番、`tag`=個体識別
（3桁数字、または未割当を示す "XXX"）。PV名だけからは判別できないため、リングの概念自体を
持たない（`flow_pv.py` は `temp_pv.py` と違い ring フィールドを返さない）。

PVリストは `pv_info/FLOW_PV.csv`（1列目 PV、先頭行はヘッダ "FLOW PV"。CCG/温度計と同じ置き場・
同じ形式）。678本、リング別ファイル分割は無い（1本のリストに全セクションがまとまっている）。

**値の意味**：100% は「ある時点で取った基準流量と同じ流量が流れている」ことを表す固定の校正
基準（ユーザ確認済み）。個体差はあるが、正常な流量計はおおむね90〜150%の範囲に収まる。

### 判定の設計：実データに基づく絶対閾値方式

CCG/温度計のようなローリング基準（同じセンサの数日前を再取得して比較）を持たない理由は、
実測データ（正常参照7本 vs 異常確認済み4本、いずれも実機の記録）を分析した結果、**絶対的な
閾値だけで十分に、かつ発症の初期から検知できる**ことが分かったため：

| 指標 | 正常参照7本（実測） | 異常確認済み4本（実測） |
|---|---|---|
| 変動係数 CV（=std/mean, %） | 最大 2.18% | 8.2%〜44.2% |
| 校正基準比に対する張り付き | 無し | 1本が間欠的に0.27〜0.31%に張り付き |

異常な個体は記録期間の**最初の24hだけを切り出しても既に検知される**ことも確認済み（同じPVの
過去の健全期間との比較を必要としない）。これにより「センサ自身の過去がすでに悪化していると
検知が遅れる」というローリング基準特有の弱点（boiling frog問題）を回避できる。

判定層（`flow_judge.py`。severityは初期値。閾値は運用しながら調整可能、との了承済み）：

| 層 | 内容 | 閾値（既定） |
|---|---|---|
| `frozen` | 窓内の range(max−min) がほぼ0（値の固着） | range < 0.05% |
| `stuck_low` | 窓のロバスト中央値が校正基準100%に対して大きく低下 | sev1: <75% / sev2: <40% / sev3: <15% |
| `excess_noise` | 窓のCV（変動係数）が正常範囲を大きく超える | sev1: >4% / sev2: >8% / sev3: >15% |
| `glitch` | 単発の外れ値（ロバストσの8倍超）の割合 | 記録のみ（severityには使わない） |

severity は各層の最大値。`frozen` はCV計算自体が無意味になるため、frozen中は `excess_noise`
層をスキップする。

**実機データで判明した2点の補正**（6月の実データで `--once --end` 検証した際に発見・修正済み）：

- `frozen`（値の固着）は当初「変化が無ければ即sev3」としていたが、678本規模で実機データを見ると、
  校正基準比が**正常な範囲（例: 100〜130%）のまま**1日中まったく値が変化しないPVが16本も見つかった
  （アーカイバの記録間隔が粗い、または本当に安定しているだけの可能性が高く、必ずしも故障では
  ない）。実際の故障（`D10_02_010`）の本質は「近ゼロ値に張り付く」という**レベルの異常**であり、
  「変化が無いこと」自体は補助的な追加証拠にすぎないと判断し、**校正基準比も低い場合のみ
  `frozen_low`(sev3) とし、正常範囲での固着は `frozen_watch`(sev1・要注視のみ) に留める**よう修正した。
- `glitch`（単発の外れ値）は当初「他の層が無ければsev1」としていたが、678本×24h規模では単発の
  外れ値は統計的に一定数出るのが自然で、実機データでも28本がこれ単独でsev1になっており、
  故障の証拠としては弱すぎると判断。**severityには使わず、詳細確認用の記録（`n_glitch`/`frac`）
  としてのみ残す**よう修正した。

### 使い方

```bash
cd flow_detector
python flow_pv.py                                   # PV名パーサの自己テスト（kblogrd不要）
python flow_fetch.py selftest                        # 取得部の自己テスト（kblogrd不要）
python flow_judge.py                                 # 判定コアの自己テスト（合成データ、kblogrd不要）

# 実機での取得+判定（1回だけ・cron向け）
python flow_headless.py --once
# crontab 例（4時間おき）:
#   0 */4 * * *  cd ~/skbva_anomaly_detector/flow_detector && python flow_headless.py --once >> flow_headless.log 2>&1

# 過去の特定期間を判定窓にして確認（--end で窓の終端を指定。アーカイバ停止中の動作確認や
# 過去の既知の故障期間での閾値検証に使う。--once と併用すること）
python flow_headless.py --once --end 20260604000000

# 常駐ループとして起動する場合
python flow_headless.py --interval-hours 4

# detector_headless.py 経由（単体実行・初回投入・動作確認用）
python detector_headless.py --flow-judge
```

**ダッシュボード連携・`detector_headless.py` 連携**：CCG/IP/温度計/機器劣化検知と同じく
`detector_headless.py` の検知サイクルに相乗りする。`run_flow_judge`/`_maybe_run_flow_judge`
として配線されており、`--watch`/`--once`/引数無しの定期ループのいずれでも自動的に実行される。
既定の実行間隔は CCG/IP/温度計センサと同じ **4hごと**（`FLOW_JUDGE_EVERY_SEC`。機器劣化検知の
1日ごとより頻度は高いが、ビーム電流と無関係・直近窓だけの軽い判定なので実機負荷は小さい）。
`dashboard.py` は `flow_detector/flow_dashboard_state.json` を読み、「流量計異常検知」タブに
判定結果を表示する（リング概念が無いため、他タブと違い `rings` ラッパーの無いフラットな構造）。

**アーカイバ停止の検知**：PVリストは読めているのに全PVが `insufficient_data`（有効データが
1点も取れていない）なら、個々のセンサ故障ではなく**アーカイバ(kblog)自体がデータ取得を停止
している**と判断し、`archiver_stopped: true` としてJSONに記録する（温度計異常検知・機器劣化
検知と同じ判定ロジック）。ダッシュボードはこのフラグを見て、警告バナーを表示する
（シャットダウン期間中に「異常なし」と誤解されないようにするため）。

**クリック展開プロット**：異常（sev≥1）のPVには、判定窓の間引き時系列（`plot: {t, v}`、
400点以下）がJSONに埋め込まれ、ダッシュボードで行をクリックすると流量[%]の時系列グラフが
表示される（温度計異常検知と同じ方式。judge時点のデータをそのまま使うため、クリック時の
kblogrd再取得は発生しない）。値の固着・校正基準比の低下・指示値不安定のどのパターンかは
グラフを見れば一目で分かる。

### 要実機確認の項目

- **kblogrd のログ群名**：`flow_fetch.py` の `LOG_GROUP` は `"VA/VAFlow"`（実機確認済み）。
  環境変数 `FLOW_KBLOGRD_LOG_GROUP` でコード変更なしに一時的な上書きもできる。
- **サンプリング間隔**：実アーカイブは5秒刻みだが、678本×24hのデータ量を抑えるため取得の既定
  間隔は30秒にしている（`flow_fetch.DEFAULT_INTERVAL`）。閾値は実測5秒生データで較正したため、
  間引きの影響（ノイズの見え方が多少変わる可能性）が気になる場合は `--interval 5` で実測条件に
  合わせて確認できる。

---

## 16. 参考文献

検知の方式（回帰モデル・特徴量・ニューラルネットによる正常/異常判定と原因推定）は、次の論文に基づいています。

- Y. Suetsugu, "Machine-learning-based pressure-anomaly detection system for SuperKEKB accelerator,"
  Phys. Rev. Accel. Beams **27**, 063201 (2024).
  DOI: [10.1103/PhysRevAccelBeams.27.063201](https://doi.org/10.1103/PhysRevAccelBeams.27.063201)

機械学習まわりの用語・手法の背景説明は、次の書籍を参考にしています。

- 伊藤真 著、『Pythonで動かして学ぶ！新しい機械学習の教科書 第３版』、翔泳社。


---

## 17. 用語(Glossary)

このプロジェクトで出てくる用語の早見表。**機械学習特有の語はほとんど無く**、大半は統計・対数まわりの一般用語か、このプログラム内で便宜的に付けた名前です。とくに「層」はニューラルネットの層とは別物（下記）。

### 一般用語（その分野で広く通じる）

- **dex** — 対数の世界での「1桁」。"decimal exponent" の略。`+1 dex = 10倍`、`+2 dex = 100倍`、`-1 dex = 1/10`。イオンポンプ電流は健全〜破損で何桁も動くので、差ではなく「何桁ずれたか(dex)」で測る。例：1e-6 A → 1e-4 A は「+2 dex」。
- **パーセンタイル（p50 / p95）** — データを小さい順に並べたときの位置。`p50` = 真ん中の値（＝**中央値 median**）、`p95` = 下から95%目の値。最大値だと一瞬のスパイクに振られるので、「正常の上端」には p95 を使う。
- **中央値(median)** — 並べて真ん中の値。平均と違い、極端な外れ値に強い（ロバスト）。電流のように外れ値の多い量に向く。
- **z スコア** — 予測値からのズレを「ばらつき(標準偏差σ)何個分か」で表した無次元量。`z=4` なら「普段のばらつきの4倍ズレている＝かなり異常」。L1 で使用。
- **ロバスト(robust)** — 外れ値があっても結果が大きく狂わない性質。median や Theil–Sen 回帰がその例。

### このプログラム独自の名前

- **学習(learn)** — ここでは深層学習のことではなく、「**基準期間（正常時）の電流を集めて、正常の範囲（p50/p95 や `I=a·P^b` モデル）を統計的に当てはめて保存する**」処理（軽い統計的学習。詳しくは末尾「線引き」参照）。`ip_judge.py learn …` がこれ。結果は `ip_models.json` に入る。
- **判定(judge)** — 学習で作った正常範囲に対し、評価したい期間の電流がどれだけ外れているかを見て severity を付ける処理。`ip_judge.py judge …`。
- **層(Layer) L0a / L0b / L1 / L2** — **ニューラルネットの層ではない**。電流の異常を別々の角度から見る独立した判定ルールに番号を振っただけで、層どうしは値を伝播しない（「検査項目」「チェック観点」と読み替えてよい）。最後に「どれか一つでも強く反応したら異常」と OR で束ねる。
  - **L0a** — 無ビーム時の電流。ビームが無ければ電流は低いはずなので、無ビームで高ければ放電を疑う（持続型＝割合、過渡スパイク型＝超過点数）。
  - **L0b** — ビームの有無を問わず、電流の中央値が「学習 p95」を何 dex 超えるか。
  - **L1** — 圧力 P と電流 I の関係（健全なら `I≈a·P^b`）からのズレを z スコアで見る。
  - **L2** — 普段連動している I-P・I-ビームの相関が崩れていないか（デカップリング）。
- **deviation_dex（学習 p95 からの逸脱）** — 判定期間の電流が、そのポンプ自身の正常上端(p95)から**何 dex 上にあるか**。中央値逸脱（持続シフト）と95%点逸脱（過渡スパイク）の大きい方を採る。acute/chronic の物差し。
- **acute / chronic（急性 / 慢性）** — 医療用語からの借用。**acute** = 自分の平常から最近大きく逸脱（今まさに何か起きた・緊急）、**chronic** = 絶対値は高いが前からその水準（要観察）。`unknown` は学習モデルが無く平常が不明。
- **severity（深刻度 0–3）** — 0=正常、1=要観察（弱い証拠）、2=注意（電流上振れ）、3=放電疑い（最重要）。
- **frac_high** — 無ビーム点のうち閾値を超えた割合（持続型放電の指標、L0a）。
- **n_excursion** — 無ビーム電流がハードシーリング(既定 1e-5 A)を超えた**点の数**（過渡スパイク型の指標、L0a）。
- **abs_hard（絶対ハードシーリング）** — 履歴に依らず「この電流を超えたら無条件で異常」とする絶対値（既定 1e-5 A）。学習窓で既に壊れていた個体を拾うバックストップ。
- **ip_trust** — そのポンプの I-P 関係が信用できるか（4U は基本 True、KEK は相関が偶発的なので慎重）。L1 下振れ・L2 圧力デカップリングの発火条件に使う。
- **supply（電源種別）** — `KEK`（KEK 製電源、低圧フロア ~1e-7 Pa）と `Agilent_4U`（4U 電源、~1e-8 Pa、I∝P 相関が綺麗）。閾値やフロアを電源種別で切り替える。
- **natural_ring / family（温度計）** — `natural_ring` は PV名から機械的に決まるリング（＝アーカイブ上どの
  `<RING>_TEMP_PV.csv` に属するか）。`ring` は `TEMP_RING_OVERRIDE.csv` 適用後の実効リング（判定・ビーム
  相関に使う）。両者を分けているのは、物理的に別リングへ移設してもPV名（取得先）は変わらないため。
  `family` は `"IR"` のとき FB_MOVE（衝突点周辺）センサであることを示す。
- **ローリング基準（rolling baseline）** — 固定モデル（一度学習して使い続ける）とは対照的に、判定のたびに
  直近数日（既定8〜5日前の3日間）を学習し直す方式。CCG・温度計は既定でこちら、イオンポンプは既定オフ
  （固定モデル方式）。回帰フィットが必要な判定は短い基準窓だと不安定になりやすい（イオンポンプの節参照）。
- **STAGE_STAGGER_SEC** — `detector_headless.py` の検知サイクル内で、CCG→イオンポンプ→温度計→
  機器劣化→流量計 の各 judge の間に挟む待機時間（既定5分）。kblogrd/EPICS への同時アクセス
  負荷を分散するため（温度計内部の LER→HER の間にも同じ待機が入る）。
- **CV（変動係数, coefficient of variation）** — std/mean。流量計異常検知で「指示値の不安定さ」を
  測る指標（%）。正常な流量計は実測で最大2.2%程度、故障個体は8%〜40%超（第II部 15 章）。
- **校正基準比（流量計）** — 流量計PVの値そのもの（%）。100% = ある時点で取った基準流量と同じ
  流量が流れている状態。正常個体でも87〜151%の個体差はあるが、大幅な低下（例: 15%未満）は
  ほぼ確実にセンサ側の異常（実流量がそこまで落ちることは運用上無いため）。
- **archiver_stopped** — 温度計・機器劣化・流量計の各判定が状態JSONに書くフラグ。PVリストは
  読めているのに全PVで有効データが1点も取れないとき true（アーカイバ(kblog)がデータ取得を
  停止している状態。シャットダウン期間など）。ダッシュボードはこれを見て警告バナーを表示する。

### 補足：CCG 側との違いと「機械学習か否か」の線引き

末次さんの **CCG 圧力検知は FNN（ニューラルネット）** を使う（§2-3, §16 参考文献）。一方 **イオンポンプ
`ip_judge` は、ニューラルネットのような複雑な機械学習モデルは使わず**、各ポンプの正常範囲を
統計的に当てはめ（ロバスト回帰・パーセンタイル）たうえで、固定ルールで外れを検知している。

ここは連続的なグラデーションで、はっきりした境界は無い点に注意（"何かのアルゴリズムで判定する=
機械学習" ではない）。目安：

- **明確に機械学習**：ニューラルネット・ランダムフォレスト・SVM・勾配ブースティング
  （多数のパラメータを最適化して汎化）。CCG 側の FNN はここ。
- **グレーゾーン（統計的学習 statistical learning）**：線形/ロバスト回帰・パーセンタイル推定。
  `ip_judge` の **learn 部分**（各ポンプの電流バンド p50/p95、`I=a·P^b` の Theil–Sen 回帰）は
  ここに当たる。「データからパラメータを当てはめる」という意味で広義の機械学習と地続き。
- **機械学習とは普通呼ばない**：固定しきい値・ルール・単純な統計量の計算。
  `ip_judge` の **判定の中心**（超過点を数える／中央値が p95 から何 dex 上か／z スコア比較）はここ。

したがって正確には、`ip_judge` は「**軽い統計的当てはめ（learn）＋ルールベース判定（judge）**」の
組み合わせ。「機械学習を一切使っていない」と言い切るより、「**ニューラルネット等の重い学習モデルは
使わず、統計的な当てはめとルールで構成**」と表現するのが正しい。

### なぜイオンポンプ側はニューラルネットを使わないのか

CCG 側（FNN）と違い、イオンポンプ `ip_judge` はあえて**ニューラルネットのような複雑な機械学習
モデルを使わず**、統計的な当てはめ＋ルールで構成している（learn 部分の `I=a·P^b` 回帰や
パーセンタイル推定は広義の統計的学習に当たる＝上の「線引き」参照）。「使えない」ではなく、今の
問題設定では軽い統計的異常検知のほうが適切で頑健なため：

- **現象が単純**：放電破損の signature は「無ビームで電流が高い」「自分の平常から何桁も跳ねる」「I-P 相関が崩れる」と 1〜2 変数で、対数で見ればほぼ単調。CCG の圧力のような多入力非線形（ビーム電流と HOM (I²/Nb)²）ではないので、非線形モデルの出番が薄い。
- **異常例が少数**：確定破損は寄せ集めても数例（D12_L23, D01_H14, D12_H05, D11_H04 …）。何百ポンプに対し異常例が一桁では教師あり NN は過学習する。正常データは大量なので「正常範囲を学んで外れを検知する」異常検知型が筋が良い。
- **個別較正と追従**：ポンプは電源種別・場所で平常電流が桁違いかつ経時変化する。今の方式は直近正常期間から p95 等を学習し直すだけで個別較正・追従できる。
- **説明可能性**：「無ビームで 2.3e-4 A、自分の平常から +2.1 dex」と理由を数値で言える。フィードスルー交換の判断根拠として、NN の「異常スコア 0.87」より強い。

将来 NN が向く場面：多変量（電流・圧力・温度・近傍ポンプ）の微妙な前兆を捉えたいとき、ラベル付き異常例が十分たまったとき。その場合も分類器より、正常だけで学習し再構成誤差で測る**オートエンコーダ等の異常検知型 NN** が少数異常例と相性が良い。acute/chronic ラベルや Normal/Abnormal 保存はその素地の蓄積にもなる。
