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
- **β の 3 つの扱い**：汎関数のデフォルト（`default`）、α + β = C の制約（`constraint`、C = 1/ε も可）、固定値（`fixed`）
- **汎関数のデフォルトを尊重**：α・β を指定しなければ IOp を書き込まず、汎関数本来の値で計算
- **Guess=Read による高速化**：2 回目以降の試行では、各状態の前回の chk を `%oldchk` + `Guess=Read` で読み込む
- **1 本のジョブで最適化全体を実行**（`--submit`）：フロントエンドで待機せず、キュー待ちも最初の 1 回だけ
- **3 状態の同時計算**：q0 / q1 / qm1 を、CPU を分けて同時に計算
- **中断からの再開**：計算済みの点は再計算せずに最適化を続ける
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

- ジョブのコア数は `nproc_per_job` × 同時に計算する状態数（`max_parallel_jobs`、最大 3）です。
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
| `molecule.gjf_file` | 基準となる gjf ファイル |
| `calculation.nproc_per_job` | 1 状態（1 計算）あたりのコア数 |
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

を加えて同じコマンドを実行します。
計算済みの点は既存の `.out` から読み込まれ、続きの点から計算が始まります。固定パラメーターや β の設定が現在と異なる結果は読み込みません。

## 注意事項

- **CPU ポイント（`--submit`）**：最適化が終わるまでジョブのコアを確保し続けるので、ある状態の計算が早く終わって他の状態を待っている間の空きコアの分もポイントが消費されます。
- **walltime（フロントエンド実行）**：完了の監視は、ジョブが始まって `.out` が現れた時点から数えて walltime + 10 分で打ち切ります。キュー待ちの時間は含みません。大きな系では `calculation.walltime` を長めに設定してください。
- **RCCS 以外での利用**：Gaussian の起動は `g16sub` と、RCCS のジョブスクリプトの形式（`GAUSS_CDEF` など）に依存しています。他の環境で使う場合は、`submit_optimization_job()` と `DeltaSCFCalculator.run_gaussian()` を書き換えてください。

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
