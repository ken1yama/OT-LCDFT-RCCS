#!/usr/bin/env python3
"""
OT-LCDFT-RCCS: 反復モード（mode: iterative）

最適チューニング（OT）と構造最適化を、ωと構造が互いに整合するまで繰り返す。
param_opt_RCCS.py から呼び出される（単独では実行しない）。

  [Step1]（任意）最初の構造最適化            iteration.initial_optimization: true
  反復 N:
    [Step2] OT → ω_N                        判定A: |ω_N − ω_{N−1}| < omega_threshold
    [Step3] ω_N で構造最適化                  判定B: RMSD < rmsd_threshold かつ
                                                     最大変位 < max_displacement_threshold
    [Step4] 新しい構造・ω_N で J を計算        判定C: |J(新しい構造) − J(前の構造)| < j_threshold
            （この点は次の反復のOTの最初の1点を兼ねる）

- 基底関数・汎関数・その他のキーワードは gjf の記述をそのまま使う（Gen/GenECP の指定も引き継ぐ）
- α・β は通常のOTと同じ優先順位（YAML > gjf > デフォルト）で決まり、全反復で共通
- 2回目以降のOTは、前回の ω を中心に ±search_range で最小値を挟み込んでから絞り込む
- 再投入すると、終わっているステップの結果を読み込んで続きから再開する
- 閾値を 0 または null にすると、その判定は使わない

判定C について:
  最小値の近くでは J ≈ J_min + k(Δω)² なので、1点の ΔJ から言える Δω は √(ΔJ/k) 程度で、
  曲率 k は系によって大きく変わる（eV² の閾値の意味が系ごとに変わる）。さらに1点だけでは、
  ω がずれたのか J_min 自体が変わったのかを区別できない。そこで既定では使わず、Step4 では
  参考として、直前のOTの曲率から見積もった Δω を表示するだけにしている。
"""

import contextlib
import io
import json
import os
import re
from pathlib import Path

import numpy as np

import param_opt_RCCS as core


DEFAULT_ITERATION = {
    'initial_optimization': False,
    'max_iterations': 5,
    'omega_threshold': 0.001,
    'rmsd_threshold': 0.01,
    'max_displacement_threshold': 0.02,
    'j_threshold': None,          # 判定C は既定では使わない（冒頭の「判定C について」を参照）
    'search_range': 0.02,
    'opt_keyword': 'Opt',
    'final_optimization': False,
}


# ---------------------------------------------------------------------------
# 構造の読み書き
# ---------------------------------------------------------------------------

def extract_last_geometry(out_file):
    """Gaussianの出力から最後の構造（Å）を取り出す

    'Input orientation' があればそれを、なければ 'Standard orientation' を使う。
    Returns: (N, 3) の numpy 配列、見つからなければ None
    """
    try:
        with open(out_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.read().split('\n')
    except OSError:
        return None

    blocks = {'Input orientation:': [], 'Standard orientation:': []}
    for i, line in enumerate(lines):
        for key in blocks:
            if key in line:
                blocks[key].append(i)

    for key in ('Input orientation:', 'Standard orientation:'):
        if not blocks[key]:
            continue
        start = blocks[key][-1] + 5  # 見出し行 → 罫線 → 2行の列名 → 罫線 → 座標
        coords = []
        for line in lines[start:]:
            if line.strip().startswith('---'):
                break
            parts = line.split()
            if len(parts) >= 6:
                try:
                    coords.append([float(parts[3]), float(parts[4]), float(parts[5])])
                except ValueError:
                    break
        if coords:
            return np.array(coords)
    return None


def kabsch_rmsd(a, b):
    """重心をそろえ、回転も合わせたうえでの RMSD と最大変位（Å）"""
    a = np.asarray(a, float) - np.mean(a, axis=0)
    b = np.asarray(b, float) - np.mean(b, axis=0)
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    a_rot = (rot @ a.T).T
    disp = np.linalg.norm(a_rot - b, axis=1)
    return float(np.sqrt(np.mean(disp ** 2))), float(np.max(disp))


def _strip_job_keywords(tokens):
    """ルート行の語から Opt / Freq を除く（構造最適化はスクリプトが付ける）"""
    return [t for t in tokens if not re.match(r'(opt|freq)\b', t, re.IGNORECASE)]


def write_template_gjf(source_gjf, dest_gjf, coords=None):
    """元のgjfをもとに、計算用のテンプレートgjfを書く

    - %chk / %oldchk は除く（スクリプトが付ける）
    - ルート行から Opt / Freq を除く
    - coords を渡せば座標をその値に置き換える（元素記号・順番は元のまま）
    - 座標より後ろ（Gen の基底関数の指定など）はそのまま残す
    """
    with open(source_gjf, 'r') as f:
        lines = f.readlines()

    route_tokens = core.read_route_line(lines).split()
    route = '#' + ' '.join(_strip_job_keywords(route_tokens))

    out = []
    i = 0
    # Link0
    while i < len(lines) and not lines[i].strip().startswith('#'):
        low = lines[i].strip().lower()
        if not (low.startswith('%chk=') or low.startswith('%oldchk=')):
            out.append(lines[i])
        i += 1
    # ルート行（複数行にまたがっていてもまとめて置き換え）
    while i < len(lines) and lines[i].strip() != '':
        i += 1
    out.append(route + '\n')

    # タイトル部分と電荷・多重度
    charge_idx = None
    for k in range(i, len(lines)):
        parts = lines[k].split()
        if len(parts) >= 2 and lines[k - 1].strip() == '':
            try:
                int(parts[0]), int(parts[1])
                if k > i + 1:  # 空行 → タイトル → 空行 → 電荷・多重度
                    charge_idx = k
                    break
            except ValueError:
                continue
    if charge_idx is None:
        raise ValueError(f"電荷・多重度の行が見つかりません: {source_gjf}")
    out.extend(lines[i:charge_idx + 1])

    # 座標
    k = charge_idx + 1
    atom_lines = []
    while k < len(lines) and lines[k].strip() != '':
        atom_lines.append(lines[k])
        k += 1
    if coords is not None:
        if len(coords) != len(atom_lines):
            raise ValueError("原子数が一致しません（構造を差し替えられません）")
        for line, (x, y, z) in zip(atom_lines, coords):
            element = line.split()[0]
            out.append(f"{element:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")
    else:
        out.extend(atom_lines)

    # 座標より後ろ
    out.extend(lines[k:])
    if not out[-1].endswith('\n'):
        out[-1] += '\n'
    out.append('\n')
    with open(dest_gjf, 'w') as f:
        f.writelines(out)


# ---------------------------------------------------------------------------
# 反復の本体
# ---------------------------------------------------------------------------

class IterativeOptimizer:
    """OTと構造最適化の繰り返し"""

    def __init__(self, config_file, launcher='g16sub'):
        self.config_file = Path(config_file).resolve()
        self.config = core.load_yaml(self.config_file)
        self.launcher = launcher
        self.base_dir = Path.cwd()

        opt = self.config.setdefault('optimization', {})
        mode = opt.get('optimization_mode', 'omega_only')
        if mode != 'omega_only':
            raise ValueError("mode: iterative は optimization_mode: omega_only のみ対応しています")

        self.it = dict(DEFAULT_ITERATION)
        self.it.update(self.config.get('iteration') or {})
        self.digits = opt.get('omega_digits', 4)
        self.omega_min = float(opt['omega_min'])
        self.omega_max = float(opt['omega_max'])

        self.source_gjf = Path(self.config['molecule']['gjf_file']).resolve()
        if not self.source_gjf.exists():
            raise FileNotFoundError(f"gjfファイルが見つかりません: {self.source_gjf}")
        self.base_name = self.source_gjf.stem
        self.nprocs = core.state_nprocs(self.config)
        self.history = []

        # α を変えるときの β の確認（Step1 の構造最適化より前に止める）
        for note in core.check_alpha_beta(self.config, self.source_gjf):
            print(note)
        with open(self.source_gjf, 'r') as f:
            route = core.read_route_line(f.readlines())
        self.beta_mode, self.beta_constraint, self.fixed_beta, _ = core.resolve_beta(opt, route)

    # ---- 判定 ----
    @staticmethod
    def _below(value, threshold):
        """閾値が 0 / null のときは判定に使わない"""
        return threshold not in (None, 0) and value < float(threshold)

    # ---- 構造最適化 ----
    def _opt_cores(self):
        """構造最適化に使うコア数とCPU（ジョブ内では全コアを1計算で使う）"""
        total = core.job_ncpus(self.config)
        cpus = None
        if self.launcher == 'direct' and os.environ.get('GAUSS_CDEF'):
            cpu_list = core.parse_cpu_list(os.environ['GAUSS_CDEF'])
            if len(cpu_list) >= total:
                cpus = core.format_cpu_list(cpu_list[:total])
        return total, cpus

    def _resolve_fixed(self):
        """構造最適化・最終gjfに使う α（と Step1 の ω）: YAML > gjf > デフォルト"""
        opt = self.config['optimization']
        with open(self.source_gjf, 'r') as f:
            gjf_omega, gjf_alpha = core.IOPParser.extract_iop_parameters(core.read_route_line(f.readlines()))
        alpha = opt.get('fixed_alpha')
        alpha = alpha if alpha is not None else gjf_alpha
        return gjf_omega, alpha

    def _calculator(self, gjf, omega, alpha, launcher, nproc, cpus=None, prev_chk=None):
        opt = self.config['optimization']
        calc = core.DeltaSCFCalculator(
            str(gjf), nproc=nproc, nproc_per_state={'opt': nproc},
            omega=omega, omega_digits=self.digits, alpha=alpha,
            epsilon=opt.get('epsilon', 1.0), beta_mode=self.beta_mode,
            beta_constraint=self.beta_constraint, fixed_beta=self.fixed_beta,
            optimization_state={'previous_chk': {'opt': prev_chk} if prev_chk else {}},
            use_guess_read=opt.get('use_guess_read', True),
            walltime=self.config.get('calculation', {}).get('walltime', '2:00:00'),
            launcher=launcher, cpu_sets={'opt': cpus} if cpus else None,
            mem_per_core_mb=int(self.config.get('calculation', {}).get('mem_per_core_mb', 1000)))
        return calc

    def run_geometry_optimization(self, step_dir, template_gjf, omega, alpha, prev_chk=None):
        """ω（と固定のα・β）で中性分子の構造を最適化する

        Returns: (最適化後の座標, エネルギー)。失敗時は例外
        """
        step_dir.mkdir(exist_ok=True)
        job_name = f"{self.base_name}_opt"
        out_file = step_dir / f"{job_name}.out"

        # 再開: 正常終了していれば読み込むだけ。途中で切れていれば最後の構造から
        if out_file.exists():
            tail = core.DeltaSCFCalculator._read_tail(str(out_file), 2048)
            coords = extract_last_geometry(out_file)
            if 'Normal termination' in tail and coords is not None:
                energy = core.GaussianResultParser.extract_energy(str(out_file))
                print(f"  既存の構造最適化の結果を使用: {out_file.relative_to(self.base_dir)}")
                return coords, energy
            if coords is not None:
                print("  途中で終わった構造最適化を、最後の構造から再開します")
                write_template_gjf(template_gjf, step_dir / 'restart_template.gjf', coords)
                template_gjf = step_dir / 'restart_template.gjf'

        nproc, cpus = self._opt_cores()
        calc = self._calculator(template_gjf, omega, alpha, self.launcher, nproc, cpus, prev_chk)
        calc.additional_keywords = list(calc.additional_keywords) + [str(self.it['opt_keyword'])]
        cwd = os.getcwd()
        try:
            os.chdir(step_dir)
            input_file = calc.create_gaussian_input(calc.charge, calc.multiplicity, job_name, 'opt')
            result = calc.run_gaussian(input_file, 'opt')
        finally:
            os.chdir(cwd)
        if result is None:
            raise RuntimeError(f"構造最適化が正常終了しませんでした: {out_file}")
        coords = extract_last_geometry(out_file)
        if coords is None:
            raise RuntimeError(f"構造最適化の出力から構造を読み取れませんでした: {out_file}")
        energy = core.GaussianResultParser.extract_energy(str(out_file))
        return coords, energy

    # ---- OT ----
    def _ot_config(self, step_dir, gjf_name):
        """各反復のOT用の設定ファイル（元のYAMLの optimization / calculation をそのまま使う）"""
        cfg = {
            'mode': 'ot',
            'molecule': {'gjf_file': gjf_name},
            'calculation': dict(self.config.get('calculation', {})),
            'optimization': dict(self.config['optimization']),
        }
        cfg['optimization']['use_existing_outputs'] = True   # 再投入時に計算済みの点を再利用
        cfg['optimization']['previous_results_file'] = None
        cfg['optimization']['keep_files'] = 'all'
        cfg['optimization'].setdefault('omega_digits', self.digits)
        path = step_dir / 'ot_config.yaml'
        with open(path, 'w') as f:
            import yaml
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        return path

    def _make_ot(self, step_dir, coords, prev_chks=None, quiet=False):
        """OTの準備。quiet: 設定の表示を省く（2回目以降は1回目と同じ内容なので）"""
        step_dir.mkdir(exist_ok=True)
        gjf_name = f"{self.base_name}.gjf"
        write_template_gjf(self.source_gjf, step_dir / gjf_name, coords)
        cfg = self._ot_config(step_dir, gjf_name)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
            optimizer = self._in_dir(step_dir, core.OmegaOptimizer, str(cfg),
                                     **{'calculation.launcher': self.launcher})
        if quiet:
            optimizer.common_info_printed = True   # 「最適化を開始します」の見出しも省く
            n_existing = len(optimizer.existing_keys)
            if n_existing:
                print(f"  （既存の計算結果 {n_existing} 点を読み込みました）")
        if prev_chks:
            # 前の反復の各状態のchkを、最初の計算のGuess=Readに使う
            for state, path in prev_chks.items():
                optimizer.optimization_state['previous_chk'].setdefault(state, path)
        return optimizer

    def _in_dir(self, step_dir, func, *args, **kwargs):
        cwd = os.getcwd()
        try:
            os.chdir(step_dir)
            return func(*args, **kwargs)
        finally:
            os.chdir(cwd)

    def _evaluate(self, optimizer, omega):
        """OTの目的関数（計算済みの点はキャッシュから）"""
        try:
            return optimizer.objective_function([round(omega, self.digits)])
        except StopIteration:
            return optimizer.evaluated_cache.get(optimizer._cache_key(round(omega, self.digits), optimizer.alpha_value))

    def run_ot_full(self, optimizer, step_dir):
        """1回目: 通常のOT（YAMLのアルゴリズム・範囲）"""
        omega, j = self._in_dir(step_dir, optimizer.optimize)
        return round(float(omega), self.digits), float(j)

    def run_ot_near(self, optimizer, step_dir, center, j_center):
        """2回目以降: center（Step4で計算済み）の両側を計算して最小値を挟み込み、放物線補間で絞り込む

        J(ω) は最小値の近くでほぼ放物線なので、3点を通る放物線の頂点を次の点にする。
        頂点が丸めた桁数で計算済みの点と同じになったら終了（それ以上は桁数の分解能を超える）。
        最小値が ±search_range の外にある場合は、全範囲の通常のOTに切り替える。
        """
        r = float(self.it['search_range'] or 0)
        res = 10 ** (-self.digits)
        a = round(max(self.omega_min, center - r), self.digits)
        c = round(min(self.omega_max, center + r), self.digits)
        if not (r > 0 and a < center < c):
            return self.run_ot_full(optimizer, step_dir)

        pts = {center: j_center}

        def f(w):
            w = round(w, self.digits)
            if w not in pts:
                pts[w] = self._in_dir(step_dir, self._evaluate, optimizer, w)
            return pts[w]

        fa, fc = f(a), f(c)
        print(f"  探索の中心 ω={center:.{self.digits}f} (J={j_center:.6f})、"
              f"両側 J({a:.{self.digits}f})={fa:.6f}, J({c:.{self.digits}f})={fc:.6f}")
        if not (j_center < fa and j_center < fc):
            print(f"  最小値が ±{r} の外にあるため、全範囲 [{self.omega_min}, {self.omega_max}] で探索します")
            return self.run_ot_full(optimizer, step_dir)

        max_evals = int(self.config['optimization'].get('n_calls', 15))
        for _ in range(max_evals):
            # 放物線で予測される J の改善が十分小さければ、頂点を1点だけ計算して終了
            step = core.parabola_step(pts, self.digits)
            if step and step[2] < core.PARABOLA_J_GAIN_TOL:
                if step[0] not in pts:
                    f(step[0])
                break
            xs = sorted(pts)
            k = min(range(len(xs)), key=lambda i: pts[xs[i]])
            if k == 0 or k == len(xs) - 1:
                break  # 挟み込みが崩れた（通常は起こらない）
            x0, x1, x2 = xs[k - 1], xs[k], xs[k + 1]
            if x2 - x0 <= 2 * res + 1e-12:
                break  # 両隣が分解能の幅まで詰まった
            y0, y1, y2 = pts[x0], pts[x1], pts[x2]
            denom = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
            if denom != 0:
                v = x1 - 0.5 * ((x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)) / denom
            else:
                v = float('nan')
            if not (x0 < v < x2):
                # 放物線が使えないときは、広い方の区間を黄金比で分ける
                v = x1 + 0.381966 * ((x2 - x1) if (x2 - x1) > (x1 - x0) else (x0 - x1))
            v = round(v, self.digits)
            if v in pts:
                break  # 頂点が計算済みの点と一致 → 分解能の範囲で収束
            f(v)

        omega = min(pts, key=pts.get)
        j = pts[omega]
        print(f"  {len(pts)}点で収束（うち1点は Step4 の点）")
        self._in_dir(step_dir, optimizer._finalize_optimization, omega, j, optimizer.alpha_value, '放物線補間（反復）')
        return round(float(omega), self.digits), float(j)

    def _best_chks(self, step_dir, optimizer, omega):
        names = optimizer._job_names(omega, optimizer.alpha_value)
        return {state: str((step_dir / name).resolve()) for state, name in names.items()
                if (step_dir / f"{name}.chk").exists()}

    @staticmethod
    def _ot_points(optimizer):
        """OTで評価した点 {ω: J}（失敗した点を除く）"""
        return {k[0]: v for k, v in optimizer.evaluated_cache.items() if k[0] is not None and v < 1000}

    @staticmethod
    def _curvature(pts):
        """最小値とその両隣の3点から、J ≈ J_min + k(ω − ω_min)² の k を見積もる"""
        xs = sorted(pts)
        if len(xs) < 3:
            return None
        i = min(range(len(xs)), key=lambda m: pts[xs[m]])
        if i == 0 or i == len(xs) - 1:
            return None
        x0, x1, x2 = xs[i - 1], xs[i], xs[i + 1]
        k = ((pts[x2] - pts[x1]) / (x2 - x1) - (pts[x1] - pts[x0]) / (x1 - x0)) / (x2 - x0)
        return k if k > 0 else None

    @staticmethod
    def _cond(label, value, unit=''):
        return f"{label}（使わない）" if value in (None, 0) else f"{label} < {value}{unit}"

    def _criterion_b(self, rmsd, max_disp):
        """判定B: 指定した方（RMSD・最大変位）をすべて満たす。両方とも 0 / null なら使わない"""
        thr = [(rmsd, self.it['rmsd_threshold']), (max_disp, self.it['max_displacement_threshold'])]
        used = [(v, t) for v, t in thr if t not in (None, 0)]
        return bool(used) and all(v < float(t) for v, t in used)

    def _check_final_geometry(self, coords, omega, prev_chks):
        """最後に構造最適化した構造で、最終の ω の J を計算する（結果の確認用、1点）"""
        print(f"\n[確認] 最終構造で ω = {omega:.{self.digits}f} の J を計算")
        step = self.base_dir / 'final_check'
        optimizer = self._make_ot(step, coords, prev_chks, quiet=True)
        j = self._in_dir(step, self._evaluate, optimizer, omega)
        self.history[-1]['j_after_opt'] = j
        self.history[-1]['final_check'] = True
        print(f"  J(OTの構造) = {self.history[-1]['j']:.6f}, J(最終構造) = {j:.6f}")

    # ---- メインループ ----
    def run(self):
        it = self.it
        fixed_omega0, alpha = self._resolve_fixed()
        print("=" * 60)
        print("反復モード（mode: iterative）: OT と構造最適化の繰り返し")
        print("=" * 60)
        print(f"分子: {self.source_gjf.name}")
        print(f"最大反復回数: {it['max_iterations']}")
        print(f"判定A: {self._cond('|Δω|', it['omega_threshold'])}")
        print(f"判定B: {self._cond('RMSD', it['rmsd_threshold'], ' Å')}, "
              f"{self._cond('最大変位', it['max_displacement_threshold'], ' Å')}")
        print(f"判定C: {self._cond('|ΔJ|', it['j_threshold'], ' eV²')}")
        print(f"2回目以降のOTの探索範囲: ±{it['search_range']}")
        print("=" * 60)

        coords = None  # None = 元のgjfの構造

        # [Step1] 最初の構造最適化
        if it['initial_optimization']:
            print("\n[Step1] 最初の構造最適化（gjfの条件）")
            step = self.base_dir / 'iter0_opt'
            step.mkdir(exist_ok=True)
            write_template_gjf(self.source_gjf, step / f"{self.base_name}.gjf")
            coords, energy = self.run_geometry_optimization(step, step / f"{self.base_name}.gjf",
                                                            fixed_omega0, alpha)
            print(f"  ✓ 完了: E = {energy:.8f} Hartree")

        verdict = None
        prev = None          # 前の反復の記録
        omega_prev = None
        prev_chks = None
        prev_points = None   # 前の反復のOTで評価した点（Step4 で Δω を見積もるため）
        last_chks = None     # 最後のOTの最適点の chk
        final_coords = coords
        for n in range(1, int(it['max_iterations']) + 1):
            print(f"\n{'=' * 60}\n反復 {n}\n{'=' * 60}")
            step_ot = self.base_dir / f"iter{n}_ot"
            geom_before = coords
            optimizer = self._make_ot(step_ot, coords, prev_chks, quiet=(n > 1))
            record = {'iteration': n}

            if n == 1:
                print("\n[Step2] OT")
                omega, j = self.run_ot_full(optimizer, step_ot)
            else:
                # [Step4]（前の反復の続き）新しい構造・前回の ω で J を計算 → 判定C
                print(f"\n[Step4] 新しい構造で ω={omega_prev:.{self.digits}f} の J を計算")
                j_after = self._in_dir(step_ot, self._evaluate, optimizer, omega_prev)
                prev['j_after_opt'] = j_after
                dj = abs(j_after - prev['j'])
                print(f"  J(前の構造)={prev['j']:.6f}, J(新しい構造)={j_after:.6f}, |ΔJ|={dj:.6f}")
                k = self._curvature(prev_points or {})
                if k:
                    est = (max(j_after - prev['j'], 0.0) / k) ** 0.5
                    prev['d_omega_estimate'] = est
                    print(f"  （参考）直前のOTの曲率 k = {k:.1f} eV²·bohr² から見積もった |Δω| ≈ {est:.{self.digits}f}")
                if self._below(dj, it['j_threshold']):
                    verdict = f"判定C（|ΔJ| = {dj:.6f} < {it['j_threshold']}）"
                    final_coords = coords
                    break
                print("\n[Step2] OT（前回の ω の周辺）")
                omega, j = self.run_ot_near(optimizer, step_ot, omega_prev, j_after)

            if j >= 1000:
                raise RuntimeError(f"反復{n}のOTで計算に失敗しました（{step_ot} を確認してください）")
            record.update({'omega': omega, 'j': j})
            prev_points = self._ot_points(optimizer)
            last_chks = self._best_chks(step_ot, optimizer, omega)
            print(f"  ✓ ω_{n} = {omega:.{self.digits}f}, J = {j:.6f}")

            if omega_prev is not None:
                d_omega = abs(omega - omega_prev)
                record['d_omega'] = d_omega
                print(f"  判定A: |Δω| = {d_omega:.{self.digits + 1}f}")
                if self._below(d_omega, it['omega_threshold']):
                    self.history.append(record)
                    verdict = f"判定A（|Δω| = {d_omega:.{self.digits + 1}f} < {it['omega_threshold']}）"
                    final_coords = coords
                    break
                # 振動の検出: ω が2つの値を行き来している
                if len(self.history) >= 2 and abs(omega - self.history[-2]['omega']) < float(it['omega_threshold'] or 0):
                    self.history.append(record)
                    verdict = "ωが振動しているため停止（収束していません）"
                    final_coords = coords
                    break

            # [Step3] 構造最適化
            print(f"\n[Step3] ω = {omega:.{self.digits}f} で構造最適化")
            step_opt = self.base_dir / f"iter{n}_opt"
            step_opt.mkdir(exist_ok=True)
            write_template_gjf(step_ot / f"{self.base_name}.gjf", step_opt / f"{self.base_name}.gjf")
            ot_chks = last_chks
            coords, energy = self.run_geometry_optimization(
                step_opt, step_opt / f"{self.base_name}.gjf", omega, optimizer.alpha_value,
                prev_chk=ot_chks.get('q0'))
            record['energy'] = energy
            before = geom_before if geom_before is not None else self._source_coords()
            rmsd, max_disp = kabsch_rmsd(before, coords)
            record.update({'rmsd': rmsd, 'max_displacement': max_disp})
            print(f"  ✓ E = {energy:.8f} Hartree")
            print(f"  判定B: RMSD = {rmsd:.4f} Å, 最大変位 = {max_disp:.4f} Å")
            self.history.append(record)
            final_coords = coords
            if self._criterion_b(rmsd, max_disp):
                verdict = f"判定B（RMSD = {rmsd:.4f} Å, 最大変位 = {max_disp:.4f} Å）"
                break

            prev, omega_prev, prev_chks = record, omega, ot_chks
        else:
            verdict = f"最大反復回数（{it['max_iterations']}）に到達（収束していません）"

        final_omega = self.history[-1]['omega']

        # 判定Aで終わった場合、最終の ω で構造を最適化し直す（任意）
        if it['final_optimization'] and verdict.startswith('判定A'):
            print(f"\n[最終] ω = {final_omega:.{self.digits}f} で構造最適化")
            step = self.base_dir / 'final_opt'
            step.mkdir(exist_ok=True)
            n = self.history[-1]['iteration']
            step_ot = self.base_dir / f"iter{n}_ot"
            write_template_gjf(step_ot / f"{self.base_name}.gjf", step / f"{self.base_name}.gjf")
            ot_chks = self._best_chks(step_ot, optimizer, final_omega)
            before = final_coords if final_coords is not None else self._source_coords()
            final_coords, energy = self.run_geometry_optimization(
                step, step / f"{self.base_name}.gjf", final_omega, optimizer.alpha_value,
                prev_chk=ot_chks.get('q0'))
            rmsd, max_disp = kabsch_rmsd(before, final_coords)
            self.history[-1].update({'energy': energy, 'rmsd': rmsd, 'max_displacement': max_disp,
                                     'final_optimization': True})
            print(f"  ✓ E = {energy:.8f} Hartree, RMSD = {rmsd:.4f} Å, 最大変位 = {max_disp:.4f} Å")

        # OTのあとに構造が変わったまま終わった場合（判定B・最大反復回数・final_optimization）は、
        # 最終構造での J を1点だけ計算して確認する
        last = self.history[-1]
        if 'energy' in last and 'j_after_opt' not in last:
            self._check_final_geometry(final_coords, final_omega, last_chks)

        self._write_outputs(verdict, final_omega, alpha, final_coords)
        return final_omega, self.history[-1]['j']

    def _source_coords(self):
        calc = core.DeltaSCFCalculator(str(self.source_gjf), nproc=1)
        return np.array([[a['x'], a['y'], a['z']] for a in calc.atoms])

    # ---- 出力 ----
    def _write_outputs(self, verdict, final_omega, alpha, final_coords):
        d = self.digits
        print(f"\n{'=' * 60}\n反復モード完了: {verdict}\n最終 ω = {final_omega:.{d}f}\n{'=' * 60}")

        # 最終構造・最終 ω の gjf（そのまま一点計算に使える。Opt/Freq は --Link1-- で分けて追加）
        final_gjf = None
        try:
            tmpl = self.base_dir / f"{self.base_name}_final_template.gjf"
            write_template_gjf(self.source_gjf, tmpl, final_coords)
            calc = self._calculator(tmpl, final_omega, alpha, 'g16sub', self.nprocs['q0'])
            calc.optimization_state = {'previous_chk': {}}
            final_gjf = calc.create_gaussian_input(calc.charge, calc.multiplicity,
                                                   f"{self.base_name}_OT_final", 'q0')
            tmpl.unlink()
            print(f"最終構造・最終 ω の入力: {final_gjf}")
        except Exception as e:
            print(f"最終gjfの作成に失敗しました: {e}")

        lines = [
            f"Executed Script: iterative_RCCS.py (OT-LCDFT-RCCS version {core.__version__})",
            "", "Iterative OT + Geometry Optimization (mode: iterative)", "=" * 70,
            f"Molecule: {self.source_gjf.name}",
            f"Result: {verdict}",
            f"Final omega: {final_omega:.{d}f}",
            f"Final input: {final_gjf}",
            "",
            f"{'Iter':<5}{'omega':<10}{'J(OT)':<12}{'J(new geom)':<14}{'dOmega':<10}"
            f"{'RMSD/A':<9}{'MaxDisp/A':<11}{'E(opt)/Hartree':<18}",
            "-" * 89,
        ]
        fmt = lambda v, s: (s.format(v) if v is not None else '-')
        for r in self.history:
            lines.append(
                f"{r['iteration']:<5}{r['omega']:<10.{d}f}{r['j']:<12.6f}"
                f"{fmt(r.get('j_after_opt'), '{:<14.6f}'):<14}{fmt(r.get('d_omega'), '{:<10.5f}'):<10}"
                f"{fmt(r.get('rmsd'), '{:<9.4f}'):<9}{fmt(r.get('max_displacement'), '{:<11.4f}'):<11}"
                f"{fmt(r.get('energy'), '{:<18.8f}'):<18}")
        last = self.history[-1] if self.history else {}
        if last.get('final_optimization') or last.get('final_check'):
            lines.append("")
        if last.get('final_optimization'):
            lines.append(f"* Iter {last['iteration']}: RMSD/MaxDisp/E(opt) are from the final "
                         "geometry optimization (final_opt/)")
        if last.get('final_check'):
            lines.append(f"* Iter {last['iteration']}: J(new geom) is J at the final geometry "
                         "and the final omega (final_check/)")
        with open(self.base_dir / f"{self.base_name}_iterative_summary.txt", 'w') as f:
            f.write("\n".join(lines) + "\n")
        with open(self.base_dir / f"{self.base_name}_iterative_summary.json", 'w') as f:
            json.dump({'verdict': verdict, 'final_omega': final_omega, 'final_input': final_gjf,
                       'iterations': self.history}, f, indent=2, ensure_ascii=False)
        self._plot()
        print(f"まとめ: {self.base_name}_iterative_summary.txt")

    def _plot(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            return
        its = [r['iteration'] for r in self.history]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(its, [r['omega'] for r in self.history], 'o-', color='tab:blue')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('ω (bohr$^{-1}$)', color='tab:blue')
        ax1.set_xticks(its)
        rm = [(r['iteration'], r['rmsd']) for r in self.history if r.get('rmsd') is not None]
        if rm:
            ax2 = ax1.twinx()
            ax2.plot([x for x, _ in rm], [y for _, y in rm], 's--', color='tab:red')
            ax2.set_ylabel('RMSD (Å)', color='tab:red')
        fig.tight_layout()
        fig.savefig(self.base_dir / f"{self.base_name}_iterative.png", dpi=150)
        plt.close(fig)


def run_iterative(config_file, launcher='g16sub'):
    """param_opt_RCCS.py の main から呼ばれる入口"""
    return IterativeOptimizer(config_file, launcher=launcher).run()
