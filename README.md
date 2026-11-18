# OT-LCDFT-RCCS

長距離補正（range-separated hybrid, LC）汎関数のパラメーター ω（および α, β）を、J 値の最小化によって系ごとに最適チューニング（optimal tuning, OT）する Python スクリプトです。
分子科学研究所 計算科学研究センター（RCCS）のスーパーコンピューターで、`g16sub` により Gaussian 16 のジョブを自動投入して最適化を進めます。

*English summary: A Python script for optimally tuning the range-separation parameter ω (and α, β) of long-range corrected DFT functionals by minimizing the J value with ΔSCF calculations in Gaussian 16. Jobs are submitted with `g16sub` on the RCCS supercomputer (Institute for Molecular Science, Japan).*

---

## 概要

各 ω（α）について、中性（N 電子）・カチオン（N−1 電子）・アニオン（N+1 電子）の 3 状態を ΔSCF 法で計算し、次の J 値を最小にするパラメーターを探索します。

```
J = (ε_HOMO(N) + IP)² + (ε_HOMO(N+1) + EA)²
IP = E(N−1) − E(N)
EA = E(N)   − E(N+1)
```

EA の条件には、中性分子の LUMO ではなくアニオンの HOMO を用います。
ファイル名では、3 つの状態を `q0`（中性）、`q1`（カチオン）、`qm1`（アニオン）と表します。

## 特徴

- **3 つの最適化モード**：ω のみ（`omega_only`）／α のみ（`alpha_only`）／ω と α を同時に（`omega_alpha`）
- **5 つの最適化アルゴリズム**：`brent`（1 次元で推奨）、`golden_section`、`grid`、`adaptive_grid`、`bayesian`
- **β の 3 つの扱い**：汎関数のデフォルト（`default`）、α + β = C の制約（`constraint`、C = 1/ε も可）、固定値（`fixed`）。α を変えるときは β の指定が必須で、α + β ≤ 1 を計算の前に確認
- **汎関数のデフォルトを尊重**：α・β を指定しなければ IOp を書き込まず、汎関数本来の値で計算
- **Guess=Read による高速化**：2 回目以降の試行では、各状態の前回の chk を `%oldchk` + `Guess=Read` で読み込む
- **1 本のジョブで最適化全体を実行**（`--submit`）：フロントエンドで待機せず、キュー待ちも最初の 1 回だけ
- **3 状態の同時計算**：q0 / q1 / qm1 を、CPU を分けて同時に計算
- **中断からの再開**：計算済みの点は再計算せずに最適化を続ける
- **J 値の計算のみのモード**（`mode: j_only`）：gjf（または YAML）のパラメーターで 3 状態を 1 回計算し、IP・EA・HOMO・J を出力
- **OT と構造最適化の繰り返し**（`mode: iterative`）：ω と構造が互いに整合するまで、OT と構造最適化を 1 本のジョブで繰り返す
- **計算ファイルの整理**（任意）：完了後に、全部残す／最適点の .out・.chk だけ残す／全部消す、を選べる

## 動作環境

- RCCS（分子研 計算科学研究センター）のスーパーコンピューター
  - Gaussian 16 と `g16sub` コマンドを使います。Gaussian の利用資格はご自身で確認してください。
- Python 3 と以下のパッケージ
  - PyYAML, NumPy, SciPy（必須）
  - scikit-optimize（`bayesian` を使う場合。インストールすると上の 3 つも一緒に入ります）
  - matplotlib（結果のグラフを出力する場合）

## セットアップ（RCCS、初回のみ）

**1. スクリプトを取得する**

```bash
cd ~
git clone https://github.com/ken1yama/OT-LCDFT-RCCS.git
```

**2. インストールスクリプトを実行する**

```bash
bash ~/OT-LCDFT-RCCS/install.sh
```

conda 環境 `otlcdft`（python, scikit-optimize, matplotlib）が作成されます。少し時間がかかります。

**3. 表示された 1 行を `~/.bashrc` に追加する**

```bash
export PATH="$HOME/OT-LCDFT-RCCS/bin:$PATH"
```

csh / tcsh の場合は `~/.cshrc` に `setenv PATH "$HOME/OT-LCDFT-RCCS/bin:${PATH}"` を追加します。
追加したらログインし直すと、`ot-g16sub` コマンドが使えるようになります。

<details>
<summary>手動でセットアップする場合</summary>

```bash
. /apl/conda/20250310/conda_init.sh          # csh / tcsh: source /apl/conda/20250310/conda_init.csh
conda create -n otlcdft python scikit-optimize matplotlib
```

環境名を `otlcdft` 以外にした場合は、`~/.bashrc` に `export OTLCDFT_ENV=<環境名>` も追加してください。

</details>

<details>
<summary>更新する場合</summary>

```bash
cd ~/OT-LCDFT-RCCS
git pull
```

`install.sh` を再実行する必要は通常ありません（必要なパッケージが変わったときは、更新内容でお知らせします）。
`examples/` の中のファイルを直接編集していると `git pull` が止まることがあるので、設定ファイルは計算用のフォルダにコピーしてから編集してください。

</details>

<details>
<summary>アンインストールする場合</summary>

```bash
# 1. ~/.bashrc（csh なら ~/.cshrc）から、追加した PATH の行を削除する
# 2. conda 環境を削除する
. /apl/conda/20250310/conda_init.sh
conda env remove -n otlcdft
# 3. スクリプトを削除する
rm -rf ~/OT-LCDFT-RCCS
```

計算用のフォルダ（gjf・out・chk・結果ファイル）は削除されません。

</details>

インストール中に `A newer version of conda exists` という警告が出ることがありますが、無視してかまいません（conda は RCCS が用意しているもので、利用者が更新する必要はありません）。

## 使い方

**1. gjf ファイルと設定ファイルを用意する**

計算用のフォルダに、基準となる gjf ファイルと YAML 設定ファイルを置きます。
例は [`examples/`](examples/) にあります。

- [`config_simple.yaml`](examples/config_simple.yaml)：ω のみを Brent 法で最適化する場合の簡易版（ふだんはこちら）
- [`config_j_only.yaml`](examples/config_j_only.yaml)：J 値を計算するだけの場合（下の「J 値の計算のみ」を参照）
- [`config_iterative.yaml`](examples/config_iterative.yaml)：OT と構造最適化を繰り返す場合（下の「OT と構造最適化の繰り返し」を参照）
- [`config_example.yaml`](examples/config_example.yaml)：全項目をコメント付きで記載した完全版

```bash
mkdir benzene && cd benzene
cp ~/OT-LCDFT-RCCS/examples/benzene.gjf .
cp ~/OT-LCDFT-RCCS/examples/config_simple.yaml config.yaml
# config.yaml を編集
```

**2. 投入する**（gjf ファイルのあるフォルダで）

```bash
ot-g16sub config.yaml
```

最適化全体が 1 本のジョブとして投入され、コマンドはすぐに終わります。あとはログアウトしてかまいません。
ジョブの中では、このスクリプトが計算ノード上で動き、q0 / q1 / qm1 の 3 つの Gaussian 計算を同時に実行しながら ω を最適化します。

```
jsub で 1 本のジョブ（例：16 コア × 3 状態 = 48 コア）
 └ 計算ノードで param_opt_RCCS.py が動く
     ├ g16 (q0,  %cpu=0-15)  ┐
     ├ g16 (q1,  %cpu=16-31) ├ 3 状態を同時に計算し、終わったら次の ω へ
     └ g16 (qm1, %cpu=32-47) ┘
```

- ジョブのコア数は、3 状態を同時に計算する場合は各状態のコア数の合計（例：16 × 3 = 48）です。
- ジョブのコア数は **64 コアまで**です。3 つの計算を 1 ノード内で走らせるため、65 コア以上（複数ノードのジョブ）にはできません。
- 開殻のイオン（q1, qm1）の計算は時間がかかるので、`nproc_per_state` で状態ごとにコア数を変えられます（例：q0 12・q1 18・qm1 18 で合計 48）。
- ジョブの walltime は `calculation.job_walltime`（既定 `'72:00:00'`）で、**最適化全体**がこの時間内に収まる必要があります。収まらなかった場合は「中断と再開」の手順で続きから再開できます。
- 進み具合は `<name>_opt.log` に出力されます（`tail -f benzene_opt.log` で確認）。

投入する前にジョブスクリプトの中身を確認したい場合は、`--prepare-only` を付けます（作成するだけで投入しません）。

```bash
ot-g16sub config.yaml --prepare-only
```

`ot-g16sub` は conda 環境を自動で有効にしてから `param_opt_RCCS.py --config config.yaml --submit` を実行するだけのコマンドです。環境を自分で有効にして、`python3 ~/OT-LCDFT-RCCS/param_opt_RCCS.py --config config.yaml --submit` と実行しても同じです。

### もう一つの実行方法：フロントエンドで実行する

conda 環境を有効にして（`. /apl/conda/20250310/conda_init.sh` → `conda activate otlcdft`）、`--submit` を付けずに実行すると、スクリプトはフロントエンドノード（ccfep）で動き、1 点（ω）ごとに 3 状態のジョブを `g16sub` で投入して完了を待ちます。
ジョブごとに必要な分だけコアを確保するので CPU ポイントの無駄は少なくなりますが、ω を変えるたびにキュー待ちが入り、スクリプトがフロントエンドで動き続けます。

```bash
# bash / zsh（ログアウトしても止まらないよう nohup でバックグラウンド実行）
nohup python3 -u ~/OT-LCDFT-RCCS/param_opt_RCCS.py --config config.yaml > opt.log 2>&1 &

# csh / tcsh
nohup python3 -u ~/OT-LCDFT-RCCS/param_opt_RCCS.py --config config.yaml >& opt.log &
```

- `-u`：出力をすぐ `opt.log` に書き込む（付けないと、ログがしばらく空のままになる）
- `> opt.log 2>&1`（csh では `>& opt.log`）：画面への出力とエラーを `opt.log` に保存
- `&`：バックグラウンドで実行し、すぐにプロンプトに戻る

```bash
tail -f opt.log                      # ログを表示し続ける（Ctrl+C で表示だけ終了）
ps -u $USER -f | grep param_opt      # 動いているか確認（2 列目がプロセス番号）
kill <プロセス番号>                    # スクリプトを止める（投入済みのジョブは止まらない）
```

フロントエンドノードでも使った CPU 時間に応じて CPU ポイントが消費されますが、このスクリプトはほとんど待機しているだけなので、消費はごくわずかです。

### その他のコマンド

```bash
python3 param_opt_RCCS.py --create-config   # 設定ファイルのひな形を作成
python3 param_opt_RCCS.py --version
python3 param_opt_RCCS.py --help
```

## gjf ファイルの準備

基準となる gjf ファイルには、**中性分子**の電荷・多重度と座標を書きます。

```
#P LC-BLYP/def2SVP

benzene

0 1
C     1.396000     0.000000     0.000000
...
```

- ルート行の汎関数・基底関数とその他のキーワードは、そのまま各計算に引き継がれます。
- `%nprocshared`（`--submit` では `%cpu`）と `%chk` はスクリプトが書き換えます。`%mem` はそのまま使われます。
- `--submit` で gjf に `%mem` がない場合は、`nproc_per_job` × `mem_per_core_mb`（既定 1000 MB）を `%mem` に書き込みます。
- ルート行に ω（`IOp(3/107,3/108)`）や α（`IOp(3/130,3/131)`）があれば、YAML で指定しない場合の固定値として使われます。
- イオンの多重度は、中性が一重項なら二重項、それ以外なら中性の多重度 − 1 になります。

## 設定ファイル（YAML）

全項目をコメント付きで記載した例が [`examples/config_example.yaml`](examples/config_example.yaml)、ω のみ・Brent 法の簡易版が [`examples/config_simple.yaml`](examples/config_simple.yaml) です。主な項目は次のとおりです。

| 項目 | 内容 |
|---|---|
| `mode` | `ot`（パラメーター最適化、既定）／`j_only`（J 値の計算のみ）／`iterative`（OT と構造最適化の繰り返し） |
| `molecule.gjf_file` | 基準となる gjf ファイル |
| `calculation.nproc_per_job` | 1 状態（1 計算）あたりのコア数 |
| `calculation.nproc_per_state` | 状態ごとのコア数（例：`{q0: 12, q1: 18, qm1: 18}`。書いた状態だけ `nproc_per_job` より優先） |
| `calculation.job_walltime` | `--submit` のジョブの walltime（最適化全体、既定 `'72:00:00'`） |
| `calculation.mem_per_core_mb` | `--submit` で gjf に `%mem` がない場合の 1 コアあたりのメモリ（既定 1000） |
| `calculation.g16sub_options` | `--submit` で g16sub に追加で渡すオプション（例：`['-q', 'H']`） |
| `calculation.mail` | `--submit` のジョブの開始・終了時にメールで通知する（`#PBS -m abe`、既定 `false`） |
| `calculation.walltime` | フロントエンド実行のときの 1 計算の walltime（既定 `'2:00:00'`） |
| `optimization.optimization_mode` | `omega_only` / `alpha_only` / `omega_alpha` |
| `optimization.optimization_algorithm` | `brent` / `golden_section` / `grid` / `adaptive_grid` / `bayesian` |
| `optimization.omega_min`, `omega_max` | ω の探索範囲 |
| `optimization.omega_digits` | ω・α・β の小数点以下の桁数（2〜4） |
| `optimization.fixed_alpha`, `fixed_omega` | 固定するパラメーター（`null` なら gjf の IOp、なければ汎関数のデフォルト） |
| `optimization.alpha_min`, `alpha_max` | α の探索範囲 |
| `optimization.beta_mode` | `default` / `constraint` / `fixed` |
| `optimization.beta_constraint` | α + β = C の C（数値、または `solvent` で 1/ε） |
| `optimization.n_calls` | 最大評価回数 |
| `optimization.j_threshold` | J 値がこの値以下になったら終了 |
| `optimization.max_parallel_jobs` | 3 状態を同時に計算する数（1 なら順番に、2 以上なら同時に） |
| `optimization.use_guess_read` | 前回の chk を Guess=Read で読み込む |
| `optimization.use_existing_outputs` | 既存の .out を読み込んで再開する |
| `optimization.keep_files` | 完了後の計算ファイル：`all`（全部残す）／`optimal`（最適点の .out・.chk だけ）／`none`（全部消す。結果ファイルとグラフは残る） |

`optimization_algorithm` を省略すると、1 次元のモードでは `brent`、`omega_alpha` では `bayesian` になります。

Brent 法は、最小値を挟む 3 点の放物線から予測される J の改善が 1e-6 eV² 未満になったら、頂点を 1 点だけ計算して終了します。最小値の近くの J には 1e-5 eV² 程度の数値的な揺らぎがあり、それ以上絞り込んでも ω は丸めの桁の範囲でしか変わらないためです。

### α を変えるときの β

長距離の HF 交換の割合は α + β で、**1 以下**である必要があります（1 を超えると長距離の DFT 交換が負になります）。気相で長距離を正しく −1/r にするなら α + β = 1、誘電遮蔽（SRSH）なら α + β = 1/ε です。

Gaussian では α（`IOp(3/130,3/131)`）と β（`IOp(3/119,3/120)`）が独立に扱われるので、α だけ変えて β を汎関数のデフォルトのままにすると、α + β が 1 を超えることがあります（例：LC-BLYP は α = 0、β = 1 なので、α = 0.15 にすると α + β = 1.15）。
そこで、**α を指定するとき（`fixed_alpha` または gjf の IOp）や α を最適化するときは、β の指定が必須**です。`beta_mode: default` のままだと、計算を始める前（`--submit` なら投入する前）にエラーで止まります。

```yaml
optimization:
  fixed_alpha: 0.2
  beta_mode: constraint      # α + β = 1（β = 0.8）
  beta_constraint: 1.0
```

- `beta_mode: fixed` で α + β > 1 になる場合もエラーになります。
- β は gjf の `IOp(3/119,3/120)` に書いても指定できます（`beta_mode: default` のとき）。
- α を最適化するときは、`alpha_max` を `constraint` なら C、`fixed` なら 1 − β までに自動で制限します。

## J 値の計算のみ（mode: j_only）

最適化はせずに、決まったパラメーターで q0 / q1 / qm1 を 1 回ずつ計算し、J 値を求めます。最適化した ω の確認や、別の基底関数・構造での J 値の比較に使えます。

```yaml
mode: j_only
molecule:
  gjf_file: benzene.gjf
calculation:
  nproc_per_job: 8
optimization:
  max_parallel_jobs: 3
  fixed_omega: 0.17      # 省略すると gjf の IOp、それもなければ汎関数のデフォルト
```

- パラメーターの優先順位は「YAML の `fixed_omega` / `fixed_alpha` / β の設定 > gjf の IOp > 汎関数のデフォルト」です。gjf のルート行に IOp を書いておけば、YAML には何も書かなくてかまいません。
- β は gjf の `IOp(3/119=…,3/120=…)` も読み取ります（YAML の `beta_mode` が `default` のとき。他のモードでも同じです）。
- 結果は `<name>_j_only_results.txt` に出力されます（各状態の全エネルギー、IP、EA、HOMO、HOMO + IP、HOMO + EA、J）。`--submit`（`ot-g16sub`）のときのログは `<name>_j_only.log` です。
- ω・α・β は小数点以下 4 桁で扱います（`omega_digits` で変更可）。

## OT と構造最適化の繰り返し（mode: iterative）

最適な ω は構造によって変わり、構造も ω によって変わります。このモードでは、OT と構造最適化を、ω と構造が互いに整合するまで繰り返します。全体が 1 本のジョブで実行されます。

```
[Step1]（任意）最初の構造最適化              iteration.initial_optimization: true
反復 N:
  [Step2] OT → ω_N                          判定A: |ω_N − ω_{N−1}| < omega_threshold
  [Step3] ω_N で構造最適化                    判定B: RMSD < rmsd_threshold かつ 最大変位 < max_displacement_threshold
  [Step4] 新しい構造・ω_N で J を計算          判定C: |J(新しい構造) − J(前の構造)| < j_threshold（既定では使わない）
```

A・B・C のどれか 1 つを満たした時点で終了します。判定B などで、OT のあとに構造が変わったまま終わった場合は、最終構造・最終 ω の J を 1 点計算して確認します（`final_check/`、まとめの表の `J(new geom)`）。設定例は [`examples/config_iterative.yaml`](examples/config_iterative.yaml) です。

```yaml
mode: iterative
molecule:
  gjf_file: benzene.gjf
calculation:
  nproc_per_job: 8
  job_walltime: '168:00:00'
optimization:            # 1 回目の OT の設定（mode: ot と同じ）
  optimization_mode: omega_only
  optimization_algorithm: brent
  omega_min: 0.12
  omega_max: 0.5
iteration:
  max_iterations: 5
  omega_threshold: 0.001         # 判定A（bohr⁻¹）
  rmsd_threshold: 0.01           # 判定B（Å）
  max_displacement_threshold: 0.02
  j_threshold: null              # 判定C（eV²）。既定では使わない
```

| 項目（`iteration.`） | 内容（既定値） |
|---|---|
| `initial_optimization` | 最初に gjf の条件（gjf の ω、なければ汎関数のデフォルト）で構造最適化する（`false`） |
| `max_iterations` | 反復の最大回数。達したら「収束していません」として終了（`5`） |
| `omega_threshold` | 判定A：前の反復との ω の差（`0.001`） |
| `rmsd_threshold`, `max_displacement_threshold` | 判定B：前の構造との RMSD と原子の最大変位（Å、重ね合わせ後。指定した方をすべて満たしたとき）（`0.01`, `0.02`） |
| `j_threshold` | 判定C：構造が変わったことによる J の変化（eV²）（`null` = 使わない） |
| `search_range` | 2 回目以降の OT の探索幅の上限：前回の ω ± この幅（`0.02`） |
| `opt_keyword` | 構造最適化のキーワード（例：`'Opt=(CalcFC,MaxCycles=200)'`）（`Opt`） |
| `final_optimization` | 判定A で終わったとき、最終の ω でもう一度構造最適化する（`false`） |

判定の閾値を `0` または `null` にすると、その判定は使いません。

**判定C を使う場合**：判定C で終了したときに、見積もった Δω が判定A の閾値以上（または見積もれなかった）ならログとまとめに注意を出します。`j_threshold: null` にして同じフォルダで再実行すると、続きから計算します。

**判定C を既定で使わない理由**：最小値の近くでは J ≈ J_min + k(Δω)² なので、ΔJ に当たる Δω は √(ΔJ/k) で、曲率 k は系によって大きく変わります（例：ベンゼン k ≈ 30、O4IPL k ≈ 55 eV²·bohr²）。そのため eV² の閾値の意味が系ごとに変わり、`j_threshold: 0.001` は判定A（`0.001`）の数倍の Δω を許してしまいます。また 1 点の ΔJ だけでは、ω がずれたのか J_min 自体が変わったのかを区別できません。Step4 では参考として、直前の OT の曲率から見積もった Δω をログに表示します。

- **基底関数・汎関数**：gjf の記述をそのまま使います（Gen / GenECP の指定も引き継ぎます）。OT と構造最適化で同じものが使われます。gjf のルート行の `Opt` / `Freq` は取り除かれ、構造最適化のステップでだけ `opt_keyword` が付きます。
- **対応するモード**：`optimization_mode: omega_only` のみです。α・β は全反復で固定です（優先順位は「YAML > gjf の IOp > 汎関数のデフォルト」）。
- **2 回目以降の OT**：Step4 の点（新しい構造・前回の ω）を中心に両側の 2 点を計算して最小値を挟み、放物線補間で絞り込みます。両側の幅は、Step4 で見積もった Δω の 3 倍（0.005 以上、見積もりがなければ 0.01）で、挟めなければ ± `search_range`、それでも挟めなければ `omega_min`〜`omega_max` で最初から探索し直します。ω はふつう少ししか変わらないので、1 回の OT は 4〜5 点で終わります。
- **Guess=Read**：各状態の chk を、ω の試行の間だけでなく反復をまたいでも引き継ぎます。構造最適化は、直前の OT の最適点の q0 の chk を初期軌道に使います。
- **コア数**：OT は通常と同じく 3 状態を同時に計算し、構造最適化はジョブ全体のコア（例：8 × 3 = 24）で 1 本計算します。
- **ω の振動**：ω が 2 つの値を行き来している（ω_N ≈ ω_{N−2}）場合は、収束していないものとして停止します。
- **中断と再開**：同じフォルダで同じコマンド（`ot-g16sub config_iterative.yaml`）を実行すると、終わっている OT の点・構造最適化を読み込んで続きから再開します。途中で終わった構造最適化は、最後の構造から計算し直します。`use_existing_outputs` を書き換える必要はありません。
- **ファイル**：各ステップは `iter0_opt/`、`iter1_ot/`、`iter1_opt/`、`iter2_ot/`、… のフォルダで計算されます（`final_optimization` は `final_opt/`、最終構造での J の確認は `final_check/`）。再開と Guess=Read に使うため、`keep_files` は使わず、計算ファイルはすべて残します。
- **出力**：
  - `<name>_iterative_summary.txt`（`.json`）：各反復の ω、J、Step4 の J、Δω、RMSD、最大変位、構造最適化のエネルギーとステップ数。ステップ数が 1 なら、最初の構造ですでに Opt の収束条件を満たしていた（構造がほとんど動かなかった）ことを表します
  - `<name>_iterative.png`：ω と RMSD の推移
  - `<name>_OT_final.gjf`：最終構造・最終 ω の一点計算の入力（そのまま次の計算に使えます）
  - `<name>_iterative.log`：`--submit` のときの実行ログ

## 出力ファイル

ファイル名の `<name>` は gjf ファイル名（拡張子なし）です。

| ファイル | 内容 |
|---|---|
| `<name>_q0_w0.1700.gjf` / `.out` / `.chk` | 各状態・各 ω の Gaussian 入出力（α も最適化する場合は `_a0.2000` が付く） |
| `<name>_optimization_results.txt` | 最適値と、全評価点の J 値・IP・EA・HOMO の一覧 |
| `<name>_opt.log` | `--submit` のときの実行ログ |
| `<name>_<mode>_optimization.png` | J 値のグラフ |

## 中断と再開

ジョブが walltime に達した場合や、フロントエンド実行で接続が切れた・Ctrl+C で止めた場合は、同じフォルダで YAML に

```yaml
optimization:
  use_existing_outputs: true
```

を加えて同じコマンドを実行します（`mode: iterative` では書き換えは不要です）。
計算済みの点は既存の `.out` から読み込まれ、続きの点から計算が始まります。固定パラメーターや β の設定が現在と異なる結果は読み込みません。

## 注意事項

- **CPU ポイント（`--submit`）**：最適化が終わるまでジョブのコアを確保し続けるので、ある状態の計算が早く終わって他の状態を待っている間の空きコアの分もポイントが消費されます。
- **walltime（フロントエンド実行）**：完了の監視は、ジョブが始まって `.out` が現れた時点から数えて walltime + 10 分で打ち切ります。キュー待ちの時間は含みません。大きな系では `calculation.walltime` を長めに設定してください。
- **RCCS 以外での利用**：Gaussian の起動は `g16sub` と、RCCS のジョブスクリプトの形式（`GAUSS_CDEF` など）に依存しています。他の環境で使う場合は、`submit_optimization_job()` と `DeltaSCFCalculator.run_gaussian()` を書き換えてください。

## 動作確認の範囲

動作は主に**閉殻の中性有機分子**で確認しています。それ以外の系では十分にテストしていないため、思わぬ不具合が生じる可能性があります。結果ファイルの IP・EA・HOMO の値が妥当かどうかを確認してから使ってください。特に次の点に注意してください。

- **イオンの多重度**：中性が一重項ならイオンは二重項、それ以外なら「中性の多重度 − 1」に固定しています。高スピンのイオンが基底状態になる系には対応していません。
- **開殻の中性分子**：HOMO は Alpha と Beta を合わせた最高被占軌道から読み取ります。スピン汚染（⟨S²⟩ のずれ）が大きい場合や、多配置性の強い系では、ΔSCF のエネルギー・HOMO とも信頼性が下がります。
- **電荷をもつ分子、遷移金属・重元素（`Gen` / `GenECP` など）、溶媒和（`SCRF`）**：動作するように作っていますが、十分には確認していません。
- **SCF が収束しにくい系**：収束しなかった点は失敗として扱い、最適化を続けます。`SCF=XQC` などはルート行に書けばそのまま引き継がれます。

不具合を見つけた場合は、GitHub の [Issues](https://github.com/ken1yama/OT-LCDFT-RCCS/issues) で報告していただけると助かります。

## 最適化したパラメーターを使うときの注意

最適化した ω（α, β）は、ルート行の IOp で指定して使います。

| パラメーター | IOp | 書き方（例） |
|---|---|---|
| ω | `IOp(3/107=…,3/108=…)` | ω × 10⁴ を 5 桁 + `00000`（ω = 0.1700 → `0170000000`） |
| α | `IOp(3/130=…,3/131=…)` | α × 10⁴ を 5 桁（α = 0.2000 → `02000`） |
| β | `IOp(3/119=…,3/120=…)` | β × 10⁴ を 5 桁 + `00000`（β = 0.8000 → `0800000000`） |

**`Opt Freq` のように 1 つのルート行で複数の計算をまとめると、IOp は最初の計算（Opt）にしか適用されず、Freq 計算には最適化したパラメーターが反映されません**（Gaussian 16 の仕様）。
この場合は `--Link1--` で計算を分け、それぞれのルート行に IOp を書いてください。

```
%chk=mol.chk
#P LC-BLYP/def2SVP IOp(3/107=0170000000,3/108=0170000000) Opt

mol

0 1
（座標）

--Link1--
%chk=mol.chk
#P LC-BLYP/def2SVP IOp(3/107=0170000000,3/108=0170000000) Freq Geom=AllCheck Guess=Read

```

`Geom=AllCheck` で、Opt で得た構造・電荷・多重度を chk から読み込みます（2 つ目の計算にはタイトルと座標は不要です）。
出力ファイルで、2 つ目の計算のルート行にも IOp が表示されていることを確認してください。

## 謝辞

`--submit` の仕組み（`g16sub -P` で作ったジョブスクリプトの `g16` の行を差し替えて投入する方法）は、Shin-ichi Koda 氏の [rccs-dmf-g16sub](https://github.com/shin1koda/rccs-dmf-g16sub) を参考にしました。

## 参考文献

- T. Stein, L. Kronik, R. Baer, *J. Am. Chem. Soc.* **2009**, *131*, 2818.
- T. Stein, H. Eisenberg, L. Kronik, R. Baer, *Phys. Rev. Lett.* **2010**, *105*, 266802.
- L. Kronik, T. Stein, S. Refaely-Abramson, R. Baer, *J. Chem. Theory Comput.* **2012**, *8*, 1515.
- S. Refaely-Abramson, S. Sharifzadeh, M. Jain, R. Baer, J. B. Neaton, L. Kronik, *Phys. Rev. B* **2013**, *88*, 081204(R).

## ライセンス

[MIT License](LICENSE)

## 作者

Ken-ichi Yamashita
