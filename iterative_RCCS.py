#!/usr/bin/env python3
"""
OT-LCDFT-RCCS: 反復モード（mode: iterative）

最適チューニング（OT）と構造最適化を、ωと構造が互いに整合するまで繰り返す。
param_opt_RCCS.py から呼び出される（単独では実行しない）。

目標: 最終の (ω, 構造) が「構造はその ω で最適化されている」かつ「その構造で OT すると
同じ ω になる（± omega_threshold）」を満たすこと。判定は |Δω| の1つだけ。

  （任意）最初の構造最適化                  iteration.initial_optimization: true
  反復 n:
    [OT] 現在の構造で OT → ω_n             |ω_n − ω_{n−1}| < omega_threshold なら終了
    [構造最適化] ω_n で構造最適化           構造が変わらなければ終了
                                            （構造が同じなら OT の結果も同じなので、厳密に整合）

- 1回目の OT は YAML のアルゴリズム（Brent法など）で全範囲を探索する
- 2回目以降の OT: J² = t1² + t2²（t1 = ε_HOMO(N) + IP, t2 = ε_HOMO(N+1) + EA）の t1, t2 は
  最小値の近くで ω にほぼ比例し、その傾き s1, s2 は構造によってほとんど変わらない。そこで、
  前回の OT の点から傾きを求め、新しい構造で前回の ω の1点を計算して、最小値の位置を
      Δω = −(t1·s1 + t2·s2) / (s1² + s2²)        （ガウス・ニュートン法）
  で予測しながら絞り込む（ふつう 2〜3 点）。予測が使えないときは、挟み込みと放物線補間に切り替える。
  （J の変化 ΔJ から Δω を見積もる方法と違い、ずれの向きも分かり、J の最小値そのものの
  変化にも惑わされない）
- |Δω| が前の反復より小さくならなければ、収束していないものとして停止する
- 基底関数・汎関数・その他のキーワードは gjf の記述をそのまま使う（Gen/GenECP の指定も引き継ぐ）
- α・β は通常のOTと同じ優先順位（YAML > gjf > デフォルト）で決まり、全反復で共通
- 再投入すると、終わっているステップの結果を読み込んで続きから再開する
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
    'search_range': 0.02,
    'opt_keyword': 'Opt',
}

# 以前のバージョンの項目。判定を |Δω| に一本化したので使わない（書いてあっても無視する）
REMOVED_ITERATION_KEYS = ('rmsd_threshold', 'max_displacement_threshold', 'j_threshold',
                          'final_optimization')

# 構造最適化で「構造が変わらなかった」とみなす、原子の最大変位（Å）
GEOM_UNCHANGED = 1e-4


def gn_step(t, s):
    """J² = t1² + t2² を最小にする ω のずれ（ガウス・ニュートン法）

    t = (t1, t2): 現在の点の値（eV）、s = (s1, s2): ω に対する傾き（eV·bohr）
    """
    if t is None or s is None:
        return None
    den = s[0] ** 2 + s[1] ** 2
    if not den > 0:
        return None
    return -(t[0] * s[0] + t[1] * s[1]) / den


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


def count_opt_steps(out_file):
    """構造最適化の出力から、最適化のステップ数（最後の 'Step number N'）を読み取る。なければ None"""
    try:
        with open(out_file, 'r', encoding='utf-8', errors='ignore') as f:
            steps = re.findall(r'Step number\s+(\d+)\s+out of', f.read())
    except OSError:
        return None
    return int(steps[-1]) if steps else None


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

        self.digits = opt.get('omega_digits', 4)
        user_it = self.config.get('iteration') or {}
        self.it = dict(DEFAULT_ITERATION)
        self.it.update({k: v for k, v in user_it.items() if k in DEFAULT_ITERATION})
        self.notes = []         # 開始時に表示する注意（設定について）
        for k in user_it:
            if k in REMOVED_ITERATION_KEYS:
                self.notes.append(f"注意: iteration.{k} は使わなくなりました（判定は |Δω| に一本化）。"
                                  "この行は消して構いません")
            elif k not in DEFAULT_ITERATION:
                self.notes.append(f"注意: iteration.{k} は不明な項目なので無視します")
        try:
            self.threshold = float(self.it['omega_threshold'])
        except (TypeError, ValueError):
            self.threshold = 0.0
        if not self.threshold > 0:
            raise ValueError("iteration.omega_threshold は正の値にしてください（例: 0.001）")
        if self.threshold < 2 * 10 ** (-self.digits):
            self.notes.append(f"注意: omega_threshold（{self.threshold}）が ω の丸めの幅（{10 ** (-self.digits):g}）に"
                              "近いので、収束しにくくなります")
        self.omega_min = float(opt['omega_min'])
        self.omega_max = float(opt['omega_max'])

        self.source_gjf = Path(self.config['molecule']['gjf_file']).resolve()
        if not self.source_gjf.exists():
            raise FileNotFoundError(f"gjfファイルが見つかりません: {self.source_gjf}")
        self.base_name = self.source_gjf.stem
        self.nprocs = core.state_nprocs(self.config)
        self.history = []
        self.last_opt_steps = None
        self.warnings = []      # まとめにも書く注意

        # α を変えるときの β の確認（Step1 の構造最適化より前に止める）
        for note in core.check_alpha_beta(self.config, self.source_gjf):
            print(note)
        with open(self.source_gjf, 'r') as f:
            route = core.read_route_line(f.readlines())
        self.beta_mode, self.beta_constraint, self.fixed_beta, _ = core.resolve_beta(opt, route)

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
                self.last_opt_steps = count_opt_steps(out_file)
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
        self.last_opt_steps = count_opt_steps(out_file)
        return coords, energy

    def _steps_text(self):
        """構造最適化のステップ数の表示（1ステップなら、構造がほとんど動かなかったことを添える）"""
        n = self.last_opt_steps
        if n is None:
            return ""
        if n == 1:
            return "（1ステップで収束: 最初の構造ですでに Opt の収束条件を満たしていた）"
        return f"（{n} ステップ）"

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

    def _t_values(self, optimizer):
        """OTで計算した点の {ω: (t1, t2)}（t1 = ε_HOMO(N) + IP, t2 = ε_HOMO(N+1) + EA、eV）"""
        out = {}
        for r in optimizer.results_history:
            try:
                if r['omega'] is None or r['j_value'] >= 1000:
                    continue
                out[round(float(r['omega']), self.digits)] = (r['homo_ev_q0'] + r['ip_ev'],
                                                              r['homo_ev_qm1'] + r['ea_ev'])
            except (KeyError, TypeError):
                continue
        return out

    @staticmethod
    def _slopes(tvals, center, window=0.01, min_span=0.002):
        """t1, t2 の ω に対する傾き（eV·bohr）を、center の周りの点への直線の当てはめで求める

        ±window の点を使う。点の間隔が狭すぎる（min_span 未満）と数値誤差で傾きが狂うので、
        そのときは center に近い順に点を加えて広げる。求められなければ None。
        """
        xs = sorted(tvals, key=lambda x: abs(x - center))
        sel = [x for x in xs if abs(x - center) <= window]
        if len(sel) < 2 or max(sel) - min(sel) < min_span:
            sel = []
            for x in xs:
                sel.append(x)
                if len(sel) >= 2 and max(sel) - min(sel) >= min_span:
                    break
        if len(sel) < 2 or max(sel) - min(sel) < min_span:
            return None
        w = np.array(sel)
        t = np.array([tvals[x] for x in sel])
        s1 = float(np.polyfit(w, t[:, 0], 1)[0])
        s2 = float(np.polyfit(w, t[:, 1], 1)[0])
        return (s1, s2) if s1 ** 2 + s2 ** 2 > 0 else None

    def run_ot_followup(self, optimizer, step_dir, center, slopes):
        """2回目以降のOT: 新しい構造で前回の ω（center）の1点を計算し、最小値を予測しながら絞り込む

        予測には前回の OT で求めた t1, t2 の傾きを使い、この構造で 0.001 以上離れた点ができたら
        その点から傾きを求め直す。予測した点が丸めた桁数で動かなくなったら終了。
        予測が使えないとき（傾きがない、予測が前回の ω ± search_range の外、予測した点で J² が
        下がらない）は、挟み込みと放物線補間に切り替える。

        Returns: (ω, J, info)  info: j_center（新しい構造・前回の ω の J）、d_omega_pred（最初の予測）、
                               n_points（このOTで使った点の数）、method
        """
        d = self.digits
        res = 10 ** (-d)
        r = float(self.it['search_range'] or 0)
        max_evals = int(self.config['optimization'].get('n_calls', 15))
        pts = {}

        def f(w):
            w = round(w, d)
            if w not in pts:
                pts[w] = self._in_dir(step_dir, self._evaluate, optimizer, w)
            return pts[w]

        center = round(center, d)
        j_center = f(center)
        if j_center is None or j_center >= 1000:
            raise RuntimeError(f"OTの最初の点（ω = {center:.{d}f}）の計算に失敗しました（{step_dir} を確認してください）")
        info = {'j_center': j_center, 'd_omega_pred': None}
        T = self._t_values(optimizer)
        step = gn_step(T.get(center), slopes)
        if step is not None:
            info['d_omega_pred'] = step
            print(f"  予測: Δω ≈ {step:+.{d}f}（前回の OT の t1, t2 の傾きから）")

        reason = None
        w = center
        while True:
            if step is None:
                reason = "t1, t2 の傾きを求められないため"
                break
            new = round(w + step, d)
            if abs(new - center) > r or not (self.omega_min <= new <= self.omega_max):
                reason = f"予測した ω = {new:.{d}f} が前回の ω ± {r} の外にあるため"
                break
            if new == w or new in pts or len(pts) >= max_evals:
                break   # 予測が丸めた桁数で動かない → 収束
            j_new = f(new)
            if j_new is None or j_new >= 1000:
                reason = f"ω = {new:.{d}f} の計算に失敗したため"
                break
            if j_new > pts[w]:
                if abs(new - w) <= 2 * res + 1e-12:
                    break   # 丸めの桁の範囲での揺らぎ → 収束
                reason = f"予測した ω = {new:.{d}f} で J² が下がらなかったため"
                break
            # この構造で計算した点から傾きを求め直す（点の間隔が 0.001 以上のときだけ）
            T = self._t_values(optimizer)
            s_new = self._slopes({x: T[x] for x in pts if x in T}, new, min_span=0.001)
            if s_new:
                slopes = s_new
            w = new
            step = gn_step(T.get(w), slopes)

        if reason:
            print(f"  {reason}、挟み込みと放物線補間で探索します")
            omega, j, method = self._run_ot_bracket(optimizer, step_dir, f, pts, min(pts, key=pts.get))
        else:
            omega = min(pts, key=pts.get)
            j = pts[omega]
            method = 'gauss-newton'
            self._in_dir(step_dir, optimizer._finalize_optimization, omega, j, optimizer.alpha_value,
                         'ガウス・ニュートン法（反復）')
        info['method'] = method
        info['n_points'] = len(self._ot_points(optimizer)) if method == 'full' else len(pts)
        print(f"  {info['n_points']}点で収束（うち1点は、前回の ω を新しい構造で計算した点）")
        return round(float(omega), d), float(j), info

    def _run_ot_bracket(self, optimizer, step_dir, f, pts, center):
        """予測が使えないとき: center の両側を計算して最小値を挟み込み、放物線補間で絞り込む

        両側の幅は 0.01、挟めなければ ±search_range、それでも挟めなければ全範囲の通常のOT。
        J²(ω) は最小値の近くでほぼ放物線なので、3点を通る放物線の頂点を次の点にする。最小値の
        両隣の片方だけが遠い（3倍以上）ときは、先に反対側にも近い側と同じ間隔の点を置く。
        Returns: (ω, J, method)  method: 'bracket' / 'full'
        """
        d = self.digits
        res = 10 ** (-d)
        r = float(self.it['search_range'] or 0)
        widths = [w for w in (min(r, 0.01), r) if w > 0]
        bracketed = False
        for width in dict.fromkeys(widths):
            a = round(max(self.omega_min, center - width), d)
            c = round(min(self.omega_max, center + width), d)
            if not (a < center < c):
                continue
            fa, fc = f(a), f(c)
            print(f"  探索の中心 ω={center:.{d}f} (J²={pts[center]:.6f})、幅 ±{width:.{d}f}: "
                  f"J²({a:.{d}f})={fa:.6f}, J²({c:.{d}f})={fc:.6f}")
            if pts[center] < fa and pts[center] < fc:
                bracketed = True
                break
        if not bracketed:
            print(f"  最小値が ±{r} の外にあるため、全範囲 [{self.omega_min}, {self.omega_max}] で探索します")
            omega, j = self.run_ot_full(optimizer, step_dir)
            return omega, j, 'full'

        max_evals = int(self.config['optimization'].get('n_calls', 15))
        for _ in range(max_evals):
            # 放物線で予測される J の改善が十分小さければ、頂点を1点だけ計算して終了
            step = core.parabola_step(pts, d)
            if step and step[1] <= core.PARABOLA_MAX_WIDTH and step[2] < core.PARABOLA_J_GAIN_TOL:
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
            dl, dr = x1 - x0, x2 - x1
            if max(dl, dr) > 3 * min(dl, dr):
                v = round(x1 - dr if dl > dr else x1 + dl, d)
                if v not in pts:
                    f(v)
                    continue
            y0, y1, y2 = pts[x0], pts[x1], pts[x2]
            denom = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
            if denom != 0:
                v = x1 - 0.5 * ((x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)) / denom
            else:
                v = float('nan')
            if not (x0 < v < x2):
                # 放物線が使えないときは、広い方の区間を黄金比で分ける
                v = x1 + 0.381966 * ((x2 - x1) if (x2 - x1) > (x1 - x0) else (x0 - x1))
            v = round(v, d)
            if v in pts:
                break  # 頂点が計算済みの点と一致 → 分解能の範囲で収束
            f(v)

        omega = min(pts, key=pts.get)
        j = pts[omega]
        self._in_dir(step_dir, optimizer._finalize_optimization, omega, j, optimizer.alpha_value, '放物線補間（反復）')
        return omega, j, 'bracket'

    def _best_chks(self, step_dir, optimizer, omega):
        names = optimizer._job_names(omega, optimizer.alpha_value)
        return {state: str((step_dir / name).resolve()) for state, name in names.items()
                if (step_dir / f"{name}.chk").exists()}

    @staticmethod
    def _ot_points(optimizer):
        """OTで評価した点 {ω: J}（失敗した点を除く）"""
        return {k[0]: v for k, v in optimizer.evaluated_cache.items() if k[0] is not None and v < 1000}

    # ---- メインループ ----
    def run(self):
        it = self.it
        d = self.digits
        thr = self.threshold
        fixed_omega0, alpha = self._resolve_fixed()
        print("=" * 60)
        print("反復モード（mode: iterative）: OT と構造最適化の繰り返し")
        print("=" * 60)
        print(f"分子: {self.source_gjf.name}")
        print(f"最大反復回数: {it['max_iterations']}")
        print(f"収束の判定: |Δω| < {thr}（または、構造最適化で構造が変わらなかったとき）")
        print(f"2回目以降のOT: 前回の OT の点から最小値を予測して絞り込む"
              f"（予測が ±{it['search_range']} を超えるときは挟み込みで探索）")
        for note in self.notes:
            print(note)
        print("=" * 60)

        coords = None                       # None = 元のgjfの構造
        geom = {'source': f"{self.source_gjf.name}（入力の構造）", 'omega': None}

        # 最初の構造最適化（任意）
        if it['initial_optimization']:
            print("\n[最初の構造最適化]（gjfの条件）")
            step = self.base_dir / 'iter0_opt'
            step.mkdir(exist_ok=True)
            write_template_gjf(self.source_gjf, step / f"{self.base_name}.gjf")
            coords, energy = self.run_geometry_optimization(step, step / f"{self.base_name}.gjf",
                                                            fixed_omega0, alpha)
            print(f"  ✓ 完了: E = {energy:.8f} Hartree{self._steps_text()}")
            geom = {'source': "iter0_opt（gjf の条件で構造最適化）", 'omega': fixed_omega0}

        verdict = None
        converged = False
        prev = None          # 前の反復の記録
        omega_prev = None
        prev_chks = None
        slopes = None        # t1, t2 の ω に対する傾き（直前の OT の点から）
        final_coords, final_geom = coords, geom
        for n in range(1, int(it['max_iterations']) + 1):
            print(f"\n{'=' * 60}\n反復 {n}\n{'=' * 60}")
            step_ot = self.base_dir / f"iter{n}_ot"
            geom_before = coords
            optimizer = self._make_ot(step_ot, coords, prev_chks, quiet=(n > 1))
            record = {'iteration': n, 'geometry': geom['source']}

            if n == 1:
                print(f"\n[OT] 構造: {geom['source']}")
                omega, j = self.run_ot_full(optimizer, step_ot)
                record['n_points'] = len(self._ot_points(optimizer))
                record['ot_method'] = 'full'
            else:
                print(f"\n[OT] 構造: {geom['source']}（前回の ω = {omega_prev:.{d}f} から）")
                omega, j, info = self.run_ot_followup(optimizer, step_ot, omega_prev, slopes)
                prev['j_after_opt'] = info['j_center']
                record.update({'n_points': info['n_points'], 'd_omega_pred': info['d_omega_pred'],
                               'ot_method': info['method']})

            if j >= 1000:
                raise RuntimeError(f"反復{n}のOTで計算に失敗しました（{step_ot} を確認してください）")
            record.update({'omega': omega, 'j': j})
            last_chks = self._best_chks(step_ot, optimizer, omega)
            s = self._slopes(self._t_values(optimizer), omega)
            if s:
                slopes = s
            print(f"  ✓ ω_{n} = {omega:.{d}f}, J² = {j:.6f}")

            if omega_prev is not None:
                d_omega = omega - omega_prev
                record['d_omega'] = d_omega
                print(f"  Δω = {d_omega:+.{d + 1}f}（収束の条件: |Δω| < {thr}）")
                if abs(d_omega) < thr:
                    self.history.append(record)
                    verdict = f"収束（|Δω| = {abs(d_omega):.{d + 1}f} < {thr}）"
                    converged = True
                    final_coords, final_geom = coords, geom
                    break
                # |Δω| が小さくならない → 収束しない（ふつうは1回の反復で大きく縮む）
                if prev.get('d_omega') is not None and abs(d_omega) >= abs(prev['d_omega']):
                    self.history.append(record)
                    verdict = "収束していません（|Δω| が前の反復より小さくなりません）"
                    self.warnings.append(
                        f"注意: |Δω| が {abs(prev['d_omega']):.{d + 1}f} → {abs(d_omega):.{d + 1}f} と小さくなりませんでした。"
                        "構造が ω に強く応答して繰り返しが発散している、構造最適化で別の配座に移った、"
                        "omega_threshold が ω の丸めの幅に近すぎる、などが考えられます"
                        "（まとめの RMSD と各フォルダの .out を確認してください）")
                    final_coords, final_geom = coords, geom
                    break

            # 構造最適化
            print(f"\n[構造最適化] ω = {omega:.{d}f}")
            step_opt = self.base_dir / f"iter{n}_opt"
            step_opt.mkdir(exist_ok=True)
            write_template_gjf(step_ot / f"{self.base_name}.gjf", step_opt / f"{self.base_name}.gjf")
            coords, energy = self.run_geometry_optimization(
                step_opt, step_opt / f"{self.base_name}.gjf", omega, optimizer.alpha_value,
                prev_chk=last_chks.get('q0'))
            before = geom_before if geom_before is not None else self._source_coords()
            rmsd, max_disp = kabsch_rmsd(before, coords)
            record.update({'energy': energy, 'opt_steps': self.last_opt_steps,
                           'rmsd': rmsd, 'max_displacement': max_disp})
            print(f"  ✓ E = {energy:.8f} Hartree{self._steps_text()}")
            print(f"  前の構造からの変化: RMSD = {rmsd:.4f} Å, 最大変位 = {max_disp:.4f} Å")
            self.history.append(record)
            geom = {'source': f"iter{n}_opt（ω = {omega:.{d}f} で構造最適化）", 'omega': omega}
            final_coords, final_geom = coords, geom
            if max_disp < GEOM_UNCHANGED:
                verdict = "収束（構造最適化で構造が変わらなかったので、この構造で OT しても ω は同じ）"
                converged = True
                break

            prev, omega_prev, prev_chks = record, omega, last_chks
        else:
            verdict = f"収束していません（最大反復回数 {it['max_iterations']} に到達）"
            self.warnings.append(
                "注意: 最後の ω で構造最適化した構造では、まだ OT をしていません。"
                "max_iterations を増やして同じフォルダで再実行すると、続きから計算します")

        final_omega = self.history[-1]['omega']
        self._write_outputs(verdict, converged, final_omega, alpha, final_coords, final_geom)
        return final_omega, self.history[-1]['j']

    def _source_coords(self):
        calc = core.DeltaSCFCalculator(str(self.source_gjf), nproc=1)
        return np.array([[a['x'], a['y'], a['z']] for a in calc.atoms])

    # ---- 出力 ----
    def _write_outputs(self, verdict, converged, final_omega, alpha, final_coords, final_geom):
        d = self.digits
        print(f"\n{'=' * 60}\n反復モード完了: {verdict}\n最終 ω = {final_omega:.{d}f}"
              f"\n最終構造: {final_geom['source']}\n{'=' * 60}")
        for w in self.warnings:
            print(w)

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
            f"Converged: {'yes' if converged else 'no'} (|dOmega| < {self.threshold})",
            f"Final omega: {final_omega:.{d}f}",
            f"Final geometry: {final_geom['source']}",
            f"Final input: {final_gjf}",
            "",
            f"{'Iter':<5}{'omega':<9}{'J^2(OT)':<11}{'Points':<7}{'dOmega':<10}{'Pred':<10}"
            f"{'J^2(new geom)':<15}{'RMSD/A':<9}{'MaxDisp/A':<11}{'E(opt)/Hartree':<18}{'Steps':<6}",
            "-" * 111,
        ]
        fmt = lambda v, s: (s.format(v) if v is not None else '-')
        for r in self.history:
            lines.append(
                f"{r['iteration']:<5}{r['omega']:<9.{d}f}{r['j']:<11.6f}{fmt(r.get('n_points'), '{:d}'):<7}"
                f"{fmt(r.get('d_omega'), '{:+.5f}'):<10}{fmt(r.get('d_omega_pred'), '{:+.5f}'):<10}"
                f"{fmt(r.get('j_after_opt'), '{:.6f}'):<15}"
                f"{fmt(r.get('rmsd'), '{:.4f}'):<9}{fmt(r.get('max_displacement'), '{:.4f}'):<11}"
                f"{fmt(r.get('energy'), '{:.8f}'):<18}{fmt(r.get('opt_steps'), '{:d}'):<6}")
        lines += [
            "",
            "omega       : OT result on the geometry of that iteration (iteration 1: input or iter0_opt,",
            "              iteration n: the geometry optimized in iteration n-1)",
            "Points      : single-point sets (q0/q1/qm1) used in that OT",
            "dOmega      : omega(n) - omega(n-1)   /   Pred: dOmega predicted from the first point of the OT",
            "J^2(new geom): J^2 at omega(n) on the geometry optimized at omega(n)",
            "RMSD/MaxDisp: change of the geometry in the optimization of that iteration",
        ]
        if self.warnings:
            lines.append("")
            lines += [f"* {w}" for w in self.warnings]
        with open(self.base_dir / f"{self.base_name}_iterative_summary.txt", 'w') as f:
            f.write("\n".join(lines) + "\n")
        with open(self.base_dir / f"{self.base_name}_iterative_summary.json", 'w') as f:
            json.dump({'verdict': verdict, 'converged': converged, 'omega_threshold': self.threshold,
                       'final_omega': final_omega, 'final_geometry': final_geom['source'],
                       'final_input': final_gjf, 'warnings': self.warnings,
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
        dw = [(r['iteration'], abs(r['d_omega'])) for r in self.history
              if r.get('d_omega') is not None and r['d_omega'] != 0]
        if dw:
            ax2 = ax1.twinx()
            ax2.semilogy([x for x, _ in dw], [y for _, y in dw], 's--', color='tab:red')
            ax2.axhline(self.threshold, color='tab:red', lw=0.8, ls=':')
            ax2.set_ylabel('|Δω| (bohr$^{-1}$)', color='tab:red')
        fig.tight_layout()
        fig.savefig(self.base_dir / f"{self.base_name}_iterative.png", dpi=150)
        plt.close(fig)


def run_iterative(config_file, launcher='g16sub'):
    """param_opt_RCCS.py の main から呼ばれる入口"""
    return IterativeOptimizer(config_file, launcher=launcher).run()
