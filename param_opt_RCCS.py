#!/usr/bin/env python3
"""
OT-LCDFT-RCCS: 長距離補正（range-separated hybrid）汎関数のパラメーター最適化

分子科学研究所 計算科学研究センター（RCCS）のスーパーコンピューターで、
g16sub を使って Gaussian 16 のジョブを投入し、ΔSCF法で求めたIP・EAと
HOMOエネルギーから J 値を最小化して ω（および α, β）を最適チューニングする。

  J = (ε_HOMO(N) + IP)² + (ε_HOMO(N+1) + EA)²
  IP = E(N-1) - E(N),  EA = E(N) - E(N+1)

- 最適化モード: omega_only / alpha_only / omega_alpha
- 最適化アルゴリズム: brent / golden_section / grid / adaptive_grid / bayesian
- パラメーターの優先順位: YAML > gjfのIOp > 汎関数のデフォルト
- 2回目以降の試行では各状態(q0/q1/qm1)の前回chkを %oldchk + Guess=Read で再利用
- use_existing_outputs: true で中断した最適化を再開（計算済みの点は再計算しない）

Usage:
    python3 param_opt_RCCS.py --config config.yaml            # フロントエンドで実行（1点ごとにg16subで投入）
    python3 param_opt_RCCS.py --config config.yaml --submit   # 最適化全体を1本のジョブとして投入

Author : Ken-ichi Yamashita
License: MIT
"""

__version__ = "1.0.0"

import os
import subprocess
import re
import argparse
import time
import yaml
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')
import signal
import sys

try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
    import skopt
    print(f"scikit-optimize version: {skopt.__version__}")
except ImportError as e:
    print(f"scikit-optimize import error: {e}")
    SKOPT_AVAILABLE = False
except Exception as e:
    print(f"scikit-optimize error: {e}")
    SKOPT_AVAILABLE = False

try:
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Brent method will not work.")


def read_route_line(lines):
    """gjfの行リストからルートセクション（#行と続く行）を1行に連結して返す"""
    route_line = ""
    for i, line in enumerate(lines):
        if line.strip().startswith('#'):
            for current_line in (l.strip() for l in lines[i:]):
                if current_line.startswith('#'):
                    route_line += ' ' + current_line[1:].strip()
                elif current_line and not current_line.startswith('%'):
                    route_line += ' ' + current_line
                else:
                    break
            break
    return route_line


WALLTIME_MARGIN_SEC = 600  # walltime超過判定の余裕（.outのコピーバック等）


def walltime_to_seconds(walltime):
    """walltime（'H:MM:SS' 文字列、またはYAMLが60進数として読んだ整数秒）を秒に変換"""
    if isinstance(walltime, int):
        return walltime
    text = str(walltime).strip()
    if not re.fullmatch(r'\d+(:\d{1,2}){0,2}', text):
        raise ValueError(f"walltimeの形式が不正です: {walltime}（例: '2:00:00'）")
    seconds = 0
    for part in text.split(':'):
        seconds = seconds * 60 + int(part)
    return seconds


def seconds_to_walltime(seconds):
    """秒を g16sub 用の 'H:MM:SS' 文字列に変換"""
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def parse_cpu_list(text):
    """Gaussianの%CPU / GAUSS_CDEF形式（例: '0-15,32-47', '0-30/2'）をCPU番号のリストに変換"""
    cpus = []
    for part in str(text).replace(' ', '').split(','):
        if not part:
            continue
        m = re.fullmatch(r'(\d+)(?:-(\d+)(?:/(\d+))?)?', part)
        if not m:
            raise ValueError(f"CPUリストの形式を解釈できません: {text}")
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        step = int(m.group(3)) if m.group(3) else 1
        cpus.extend(range(start, end + 1, step))
    return cpus


STATES = ('q0', 'q1', 'qm1')


def state_nprocs(config):
    """各状態（q0/q1/qm1）のコア数
    
    calculation.nproc_per_state で状態ごとに指定でき（例: {q0: 12, q1: 18, qm1: 18}）、
    指定のない状態は calculation.nproc_per_job になる。
    """
    calc = config.get('calculation', {})
    default = int(calc['nproc_per_job'])
    per_state = calc.get('nproc_per_state') or {}
    unknown = set(per_state) - set(STATES)
    if unknown:
        raise ValueError(f"nproc_per_state に不明な状態があります: {sorted(unknown)}（q0, q1, qm1 のみ）")
    nprocs = {state: int(per_state.get(state, default)) for state in STATES}
    if any(n < 1 for n in nprocs.values()):
        raise ValueError(f"コア数は1以上にしてください: {nprocs}")
    return nprocs


def job_ncpus(config):
    """--submit のジョブ全体のコア数（同時実行なら各状態の合計、順番に実行なら最大値）"""
    nprocs = state_nprocs(config)
    return sum(nprocs.values()) if n_parallel_slots(config) > 1 else max(nprocs.values())


def split_cpu_sets(cdef, nprocs, parallel=True):
    """ジョブに割り当てられたCPU（GAUSS_CDEF）を各状態に割り振る
    
    parallel=True : 3状態が同時に走るので、重ならないように nprocs[状態] 個ずつ順に割り当てる
    parallel=False: 順番に走るので、各状態とも先頭から nprocs[状態] 個を使う
    
    Returns:
        dict: {状態: '%CPUに書く文字列'}。CPUが足りない場合は None
    """
    cpus = parse_cpu_list(cdef)
    need = sum(nprocs.values()) if parallel else max(nprocs.values())
    if len(cpus) < need:
        return None
    sets, pos = {}, 0
    for state in STATES:
        n = nprocs[state]
        if parallel:
            sets[state] = format_cpu_list(cpus[pos:pos + n])
            pos += n
        else:
            sets[state] = format_cpu_list(cpus[:n])
    return sets


def format_cpu_list(cpus):
    """CPU番号のリストを '0-15,32-47' の形式に変換"""
    ranges = []
    for c in cpus:
        if ranges and c == ranges[-1][1] + 1:
            ranges[-1][1] = c
        else:
            ranges.append([c, c])
    return ','.join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)


def n_parallel_slots(config):
    """3状態を同時に走らせる数（max_parallel_jobs が1なら1、2以上なら最大3）"""
    max_parallel = config.get('optimization', {}).get('max_parallel_jobs', 1) or 1
    return min(max_parallel, 3) if max_parallel > 1 else 1


def default_config():
    """デフォルト設定（--create-config のサンプルにも使用）"""
    return {
        'molecule': {
            'gjf_file': 'molecule.gjf'
        },
        'calculation': {
            'nproc_per_job': 4,
            'walltime': '2:00:00',
            'job_walltime': '72:00:00',
            'mem_per_core_mb': 1000,
            'mail': False
        },
        'optimization': {
            'optimization_mode': 'omega_only',
            'optimization_algorithm': 'brent',
            'omega_min': 0.1,
            'omega_max': 0.5,
            'omega_digits': 2,
            'fixed_omega': None,
            'fixed_alpha': None,
            'alpha_min': 0.0,
            'alpha_max': 0.5,
            'beta_mode': 'default',
            'beta_constraint': None,
            'fixed_beta': None,
            'epsilon': 1.0,
            'n_calls': 15,
            'n_initial_points': 8,
            'bayesian_acq_func': 'gp_hedge',
            'j_threshold': None,
            'max_parallel_jobs': 1,
            'previous_results_file': None,
            'use_existing_outputs': False,
            'use_guess_read': True,
            'keep_files': 'all'
        }
    }


class IOPParser:
    """gjfファイルからIOPパラメータを解析するクラス"""

    @staticmethod
    def extract_iop_parameters(route_line):
        """ルートラインからIOP指定のω、αパラメータを抽出"""
        omega_from_iop = None
        alpha_from_iop = None

        # ω パラメータの抽出 IOp(3/107=MMMMMNNNNN,3/108=MMMMMNNNNN)
        # ω = MMMMM / 10000（例: 0181800000 -> 0.1818）
        omega_pattern = r'IOp\(3/107=(\d{10}),3/108=\1\)'
        omega_match = re.search(omega_pattern, route_line, re.IGNORECASE)
        if omega_match:
            omega_from_iop = int(omega_match.group(1)[:5]) / 10000.0

        # α パラメータの抽出 IOp(3/130=XXXXX,3/131=XXXXX)
        alpha_pattern = r'IOp\(3/130=(\d{5}),3/131=\1\)'
        alpha_match = re.search(alpha_pattern, route_line, re.IGNORECASE)
        if alpha_match:
            alpha_str = alpha_match.group(1)
            # 5桁の数値を10000で割る
            # 例: 02000 -> 0.2
            alpha_from_iop = int(alpha_str) / 10000.0
        
        return omega_from_iop, alpha_from_iop


class GaussianResultParser:
    """Gaussian計算結果の解析を担当するクラス"""
    
    @staticmethod
    def extract_energy(output_file):
        """SCFエネルギーを抽出"""
        if not os.path.exists(output_file):
            return None
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        pattern = r"SCF Done:.*E\(.*\)\s*=\s*([-\d.]+)"
        matches = re.findall(pattern, content)
        
        if matches:
            return float(matches[-1])
        return None
    
    @staticmethod
    def extract_homo_lumo(output_file):
        """HOMO/LUMOエネルギーを抽出（最後に出力された軌道エネルギーを使用）
        
        開殻（非制限）計算では、AlphaとBetaを合わせて
        HOMO = 最も高い被占軌道、LUMO = 最も低い空軌道 とする（IP定理の最高被占軌道）。
        閉殻（制限）計算ではAlphaのみが出力されるので、Alphaだけの値になる。
        """
        if not os.path.exists(output_file):
            return None, None
        
        with open(output_file, 'r') as f:
            lines = f.read().split('\n')
        
        eig = re.compile(r'\s+(Alpha|Beta)\s+(occ|virt)\.\s+eigenvalues\s*--(.*)')
        
        # 最後の軌道エネルギー出力ブロック（連続する eigenvalues 行）を探す
        end = None
        for k in range(len(lines) - 1, -1, -1):
            if eig.match(lines[k]):
                end = k
                break
        if end is None:
            return None, None
        start = end
        while start > 0 and eig.match(lines[start - 1]):
            start -= 1
        
        number = r'-?\d+\.\d+'  # 桁あふれで数値がつながっている場合にも対応
        occ, virt = [], []
        for line in lines[start:end + 1]:
            m = eig.match(line)
            values = [float(x) for x in re.findall(number, m.group(3))]
            (occ if m.group(2) == 'occ' else virt).extend(values)
        
        homo_energy = max(occ) if occ else None
        lumo_energy = min(virt) if virt else None
        return homo_energy, lumo_energy
    
    @staticmethod
    def extract_calculation_info(output_file):
        """出力ファイルから計算設定情報を抽出"""
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            
            # 汎関数の確認
            method_pattern = r"#.*?(\w+-?\w*)/.*?\n"
            method_match = re.search(method_pattern, content, re.IGNORECASE)
            method = method_match.group(1).lower() if method_match else None
            
            # 基底関数の確認
            basis_pattern = r"#.*?/([^\s]+)"
            basis_match = re.search(basis_pattern, content)
            basis = basis_match.group(1).lower() if basis_match else None
            
            # 電荷・多重度の確認
            charge_mult_pattern = r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)"
            charge_mult_match = re.search(charge_mult_pattern, content)
            charge = int(charge_mult_match.group(1)) if charge_mult_match else None
            multiplicity = int(charge_mult_match.group(2)) if charge_mult_match else None
            
            return {
                'method': method,
                'basis': basis,
                'charge': charge,
                'multiplicity': multiplicity
            }
        except:
            return None


class JValueCalculator:
    """J値計算を担当するクラス"""
    
    HARTREE_TO_EV = 27.2114
    
    @staticmethod
    def calculate_j_value(energies, homo_energy_q0, homo_energy_qm1):
        """エネルギー辞書からJ値を計算（anionのHOMOを使用）"""
        ip_hartree = energies['q1'] - energies['q0']
        ea_hartree = energies['q0'] - energies['qm1']
        
        ip_ev = ip_hartree * JValueCalculator.HARTREE_TO_EV
        ea_ev = ea_hartree * JValueCalculator.HARTREE_TO_EV
        homo_ev_q0 = homo_energy_q0 * JValueCalculator.HARTREE_TO_EV
        homo_ev_qm1 = homo_energy_qm1 * JValueCalculator.HARTREE_TO_EV
        
        # J値の計算: q0のHOMOとIP、qm1のHOMOとEAを比較
        j_value = (homo_ev_q0 + ip_ev)**2 + (homo_ev_qm1 + ea_ev)**2
        
        return {
            'j_value': j_value,
            'ip_ev': ip_ev,
            'ea_ev': ea_ev,
            'homo_ev_q0': homo_ev_q0,
            'homo_ev_qm1': homo_ev_qm1
        }


class FileNameGenerator:
    """ファイル名生成を担当するクラス"""
    
    @staticmethod
    def generate_job_name(base_name, state, omega=None, alpha=None, omega_digits=3):
        """ジョブ名を生成"""
        job_name = f"{base_name}_{state}"
        
        if omega is not None:
            omega_format = f"{{:.{omega_digits}f}}"
            job_name += f"_w{omega_format.format(omega)}"
        
        if alpha is not None:
            alpha_format = f"{{:.{omega_digits}f}}"
            job_name += f"_a{alpha_format.format(alpha)}"
        
        return job_name
    
    @staticmethod
    def extract_parameters_from_filename(filename, optimization_mode):
        """ファイル名からω、αパラメーターを抽出"""
        omega_pattern = r"_w(\d+\.?\d*)"
        alpha_pattern = r"_a(\d+\.?\d*)"
        
        omega_match = re.search(omega_pattern, filename)
        alpha_match = re.search(alpha_pattern, filename)
        
        if optimization_mode == 'omega_only':
            # ω最適化：ωが必須、αはオプション
            if omega_match:
                omega = float(omega_match.group(1))
                alpha = float(alpha_match.group(1)) if alpha_match else None
                return omega, alpha
        elif optimization_mode == 'alpha_only':
            # α最適化：αが必須、ωはオプション
            if alpha_match:
                alpha = float(alpha_match.group(1))
                omega = float(omega_match.group(1)) if omega_match else None
                return omega, alpha
        elif optimization_mode == 'omega_alpha':
            # ω+α最適化：両方が必須
            if omega_match and alpha_match:
                omega = float(omega_match.group(1))
                alpha = float(alpha_match.group(1))
                return omega, alpha
        
        return None, None


class DeltaSCFCalculator:
    def __init__(self, gjf_file, nproc=4, omega=None, omega_digits=3, alpha=None, 
                 epsilon=1.0, beta_mode='default', beta_constraint=None, fixed_beta=None,
                 optimization_state=None, use_guess_read=True, walltime='2:00:00',
                 launcher='g16sub', cpu_sets=None, mem_per_core_mb=1000, nproc_per_state=None):
        """
        Parameters:
        -----------
        gjf_file : str
            入力gjfファイルのパス
        nproc : int
            使用するプロセッサ数
        omega : float
            ωパラメーター
        omega_digits : int
            ωの小数点以下桁数 (2-4)
        alpha : float
            αパラメーター
        epsilon : float
            εパラメーター（比誘電率、1以上、beta_constraint='solvent'時のみ使用）
        beta_mode : str
            βの扱い方 ('default', 'constraint', 'fixed')
        beta_constraint : float or str
            beta_mode='constraint'時の制約条件（数値または'solvent'）
        fixed_beta : float
            beta_mode='fixed'時のβ固定値
        optimization_state : dict
            最適化状態を追跡するための辞書
        walltime : str or int
            g16subに渡すwalltime（'H:MM:SS'、または秒）
        launcher : str
            'g16sub'（1点ごとにg16subで投入）または 'direct'（ジョブ内でg16を直接実行）
        cpu_sets : dict
            launcher='direct' のとき、各状態に割り当てるCPU（%CPUの文字列）
        mem_per_core_mb : int
            launcher='direct' でgjfに%memがない場合の、1コアあたりのメモリ（MB）
        nproc_per_state : dict
            状態ごとのコア数（例: {'q0': 12, 'q1': 18, 'qm1': 18}）。ない状態は nproc
        """
        self.gjf_file = gjf_file
        self.nproc = nproc
        self.nproc_per_state = nproc_per_state or {}
        self.omega_digits = omega_digits
        self.optimization_state = optimization_state if optimization_state is not None else {}
        self.use_guess_read = use_guess_read
        walltime_sec = walltime_to_seconds(walltime)
        self.walltime = seconds_to_walltime(walltime_sec)
        # ジョブ開始（.out出現）からのタイムアウト = walltime + 余裕
        self.job_timeout = walltime_sec + WALLTIME_MARGIN_SEC
        self.launcher = launcher
        self.cpu_sets = cpu_sets or {}
        self.mem_per_core_mb = mem_per_core_mb
        
        # β関連パラメータの保存
        self.beta_mode = beta_mode
        self.beta_constraint = beta_constraint
        self.fixed_beta = fixed_beta
        self.epsilon = round(epsilon, self.omega_digits) if epsilon >= 1.0 else 1.0
        
        # ωとαを指定桁数で処理
        self.omega = round(omega, self.omega_digits) if omega is not None else None
        self.alpha = round(alpha, self.omega_digits) if alpha is not None else None
        
        # βの決定
        self.beta = self._determine_beta()
            
        self.base_name = Path(gjf_file).stem
        
        # gjfファイルから設定を読み取る
        parsed_data = self.parse_gjf_file()
        self.method = parsed_data['method']
        self.basis = parsed_data['basis']
        self.mem = parsed_data['mem']
        self.atoms = parsed_data['atoms']
        self.charge = parsed_data['charge']
        self.multiplicity = parsed_data['multiplicity']
        self.additional_keywords = parsed_data['additional_keywords']
        self.has_scrf = parsed_data['has_scrf']
        self.post_geometry_content = parsed_data['post_geometry_content']
        
    
    def _determine_beta(self):
        """βの値を決定"""
        if self.alpha is None:
            # αが指定されていない場合、βも指定しない
            return None
        
        if self.beta_mode == 'default':
            # ①デフォルト：βのIOP指定なし（Gaussianのデフォルトを使用）
            return None
        
        elif self.beta_mode == 'constraint':
            # ②制約条件：α + β = C
            if self.beta_constraint is None:
                raise ValueError("beta_mode='constraint'の場合、beta_constraintの指定が必要です")
            
            # Cの決定
            if isinstance(self.beta_constraint, str) and self.beta_constraint.lower() == 'solvent':
                # 'solvent'の場合のみεを使用
                if self.epsilon < 1.0:
                    raise ValueError("ε（比誘電率）は1以上の正数である必要があります")
                C = 1.0 / self.epsilon
            else:
                # 数値の場合
                C = float(self.beta_constraint)
            
            # Cの検証
            if not (0 < C <= 1.0):
                raise ValueError(f"beta_constraint（C）は0 < C ≤ 1 である必要があります（現在の値: {C}）")
            
            # β = C - α
            beta = round(C - self.alpha, self.omega_digits)
            
            # βの検証
            if beta < 0:
                raise ValueError(f"計算されたβ（{beta}）が負の値です。α={self.alpha}, C={C}")
            
            return beta
        
        elif self.beta_mode == 'fixed':
            # ③固定値：βを指定値で固定
            if self.fixed_beta is None:
                raise ValueError("beta_mode='fixed'の場合、fixed_betaの指定が必要です")
            return round(self.fixed_beta, self.omega_digits)
        
        else:
            raise ValueError(f"不正なbeta_mode: {self.beta_mode}。'default', 'constraint', 'fixed'のいずれかを指定してください")


    def parse_gjf_file(self):
        """gjfファイルから計算設定と分子構造を読み取る"""
        with open(self.gjf_file, 'r') as f:
            lines = f.readlines()
        
        mem = None
        method = None
        basis = None
        additional_keywords = []
        has_scrf = False

        # Link0部分（ルート行より前）から%memを取得
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                break
            if line.lower().startswith('%mem='):
                mem = line.split('=', 1)[1]

        # ルートラインから情報を抽出
        route_parts = read_route_line(lines).split()
        
        # method/basisの抽出
        for part in route_parts:
            if '/' in part and not part.startswith('IOp'):
                method_basis = part.split('/')
                if len(method_basis) >= 2:
                    method = method_basis[0]
                    basis = '/'.join(method_basis[1:])
                    break
        
        # 追加キーワードの抽出（method/basis以外のすべて）
        for part in route_parts:
            if part and part != method + '/' + basis and not part.upper() in ['P', 'SP', 'T']:
                additional_keywords.append(part)
                if part.lower().startswith('scrf='):
                    has_scrf = True
        
        if method is None or basis is None:
            raise ValueError("gjfファイルからmethod/basisを抽出できませんでした")
        
        # 分子構造の部分を探す
        geometry_start = None
        charge = None
        multiplicity = None
        
        for i, line in enumerate(lines):
            if line.strip() == '' and i > 0:
                if i + 1 < len(lines):
                    try:
                        charge_mult_line = lines[i + 1].strip()
                        if charge_mult_line:
                            charge, multiplicity = map(int, charge_mult_line.split()[:2])
                            geometry_start = i + 2
                            break
                    except:
                        continue
        
        if geometry_start is None or charge is None or multiplicity is None:
            raise ValueError("gjfファイルから分子構造または電荷・多重度を読み取れませんでした")
        
        # 座標の読み取り
        atoms = []
        geometry_end = None
        for i in range(geometry_start, len(lines)):
            line = lines[i].strip()
            if line == '':
                geometry_end = i
                break
            
            parts = line.split()
            if len(parts) >= 4:
                try:
                    atoms.append({
                        'element': parts[0],
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'z': float(parts[3])
                    })
                except ValueError:
                    geometry_end = i
                    break
            else:
                geometry_end = i
                break
        
        # 座標以降のコンテンツを保存
        post_geometry_content = ""
        if geometry_end is not None:
            for line in lines[geometry_end:]:
                post_geometry_content += line
        
        return {
            'method': method,
            'basis': basis,
            'mem': mem,
            'atoms': atoms,
            'charge': charge,
            'multiplicity': multiplicity,
            'additional_keywords': additional_keywords,
            'has_scrf': has_scrf,
            'post_geometry_content': post_geometry_content
        }
    
    def create_gaussian_input(self, charge, multiplicity, job_name, job_suffix):
        """Gaussianの入力ファイルを作成
        
        chkファイル名はjob_name（ω/α値を含むユニーク名）を使用。
        2回目以降の試行では、同一状態の前回試行のchkをoldchkで指定し、
        Guess=Readにより波動関数を再利用してSCF収束を高速化する。
        """
        input_file = f"{job_name}.gjf"
        chk_name = job_name  # 今回の計算のchk名（ω/α値を含むユニーク名）

        # IOPオプションの準備
        iop_options = []
        
        # ωのIOPオプション
        if self.omega is not None:
            factor = 10 ** self.omega_digits
            omega_int = round(self.omega * factor)
            omega_str = f"{omega_int:0{self.omega_digits + 1}d}" + "0" * (10 - self.omega_digits - 1)
            iop_options.append(f"IOp(3/107={omega_str},3/108={omega_str})")
        
        # αのIOPオプション
        if self.alpha is not None:
            alpha_int = round(self.alpha * 10000)
            alpha_str = f"{alpha_int:05d}"
            iop_options.append(f"IOp(3/130={alpha_str},3/131={alpha_str})")
            
            # βの処理（Noneでない場合のみIOP指定）
            if self.beta is not None:
                beta_int = round(self.beta * 10000)
                beta_str = f"{beta_int:05d}00000"
                iop_options.append(f"IOp(3/119={beta_str},3/120={beta_str})")
        
        # IOPオプションを結合
        iop_option_str = " " + " ".join(iop_options) if iop_options else ""
        
        # 追加キーワードの処理（ω/α/β関連の既存IOPは除外、大文字小文字を区別しない）
        additional_str = ""
        for keyword in self.additional_keywords:
            if not keyword.lower().startswith(('iop(3/10', 'iop(3/11', 'iop(3/13')):
                additional_str += f" {keyword}"

        # 同じ状態(q0/q1/qm1)の前回成功したchkがあれば %oldchk + Guess=Read
        prev_chk = self.optimization_state.get('previous_chk', {}).get(job_suffix)
        use_guess = bool(prev_chk) and self.use_guess_read
        guess_str = " Guess=Read" if use_guess else ""

        with open(input_file, 'w') as f:
            if self.launcher == 'direct' and job_suffix in self.cpu_sets:
                # ジョブ内で3状態を同時に走らせるので、使うCPUを状態ごとに分ける
                f.write(f"%cpu={self.cpu_sets[job_suffix]}\n")
            else:
                f.write(f"%nprocshared={self._nproc(job_suffix)}\n")
            if self.mem:
                f.write(f"%mem={self.mem}\n")
            elif self.launcher == 'direct':
                # g16subを通さないので%memを自分で指定する
                f.write(f"%mem={self._nproc(job_suffix) * self.mem_per_core_mb}MB\n")
            if use_guess:
                f.write(f"%oldchk={prev_chk}.chk\n")  # %chkより前に記述
            f.write(f"%chk={chk_name}.chk\n")
            f.write(f"#P {self.method}/{self.basis}{iop_option_str}{additional_str}{guess_str}\n")
            f.write("\n")
            f.write(f"{job_name} calculation\n")
            f.write("\n")
            f.write(f"{charge} {multiplicity}\n")
            
            for atom in self.atoms:
                f.write(f"{atom['element']:2s} {atom['x']:12.6f} {atom['y']:12.6f} {atom['z']:12.6f}\n")
            
            if self.post_geometry_content.strip():
                if not self.post_geometry_content.startswith('\n'):
                    f.write("\n")
                f.write(self.post_geometry_content)
            else:
                f.write("\n")
        
        return input_file
    
    def monitor_job_completion(self, gjf_path, check_interval=15):
        """計算完了を監視
        
        .outファイル内に 'Normal termination' または 'Error termination' が
        出現するまでポーリングする。ファイルサイズではなく内容で判断するため、
        コピー途中やバッファリング中の誤検出を防ぐ。
        
        タイムアウトは.outファイルが出現した時点（ジョブ開始）から数えるので、
        キュー待ちの時間は含まれない。walltime超過で打ち切られたジョブは
        walltime + 余裕 を過ぎた時点で失敗と判定する。
        
        Returns:
            str: 出力ファイルパス（成功・失敗問わず完了時）
            None: タイムアウト時
        """
        base_name = gjf_path.stem
        out_file = str(gjf_path.parent / f"{base_name}.out")
        started_at = None
        
        while True:
            if os.path.exists(out_file):
                if started_at is None:
                    started_at = time.time()
                tail = self._read_tail(out_file, 1024)
                if 'Normal termination' in tail or 'Error termination' in tail:
                    return out_file
                if time.time() - started_at > self.job_timeout:
                    print(f"    タイムアウト（walltime {self.walltime} + 余裕）を超過しました: {base_name}")
                    return None
            time.sleep(check_interval)
    
    @staticmethod
    def _read_tail(file_path, nbytes):
        """ファイル末尾nbytesを文字列で返す（存在しない・読めない場合は空文字列）"""
        try:
            with open(file_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - nbytes))
                return f.read().decode('utf-8', errors='ignore')
        except OSError:
            return ""

    def check_calculation_success(self, out_file_path):
        """計算成功を判定（.outファイル末尾のNormal terminationを確認）"""
        if out_file_path is None:
            return False

        tail = self._read_tail(out_file_path, 2048)
        if 'Normal termination' in tail:
            return True

        # Error terminationの場合はエラー内容を表示
        lines = tail.split('\n')
        for i, line in enumerate(lines):
            if 'Error termination' in line:
                error_context = '\n'.join(lines[max(0, i - 3):i + 1])
                print(f"    Gaussian Error: {error_context.strip()}")
                break
        return False
    
    def run_gaussian(self, input_file, job_suffix=None):
        """Gaussianを実行（g16subで投入、またはジョブ内でg16を直接実行）"""
        gjf_path = Path(input_file)
        base_name = gjf_path.stem
        
        # 同名の古い.out（以前の失敗した計算など）が残っていると完了判定を誤るため退避
        old_out = gjf_path.parent / f"{base_name}.out"
        if old_out.exists():
            os.replace(old_out, gjf_path.parent / f"{base_name}.out.old")
        
        if self.launcher == 'direct':
            return self._run_g16_direct(gjf_path, job_suffix)
        
        command = ["g16sub", "-np", str(self._nproc(job_suffix)), "--walltime", self.walltime, input_file]
        
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=gjf_path.parent
            )
            
            stdout, stderr = process.communicate()
            g16sub_return_code = process.returncode
            
            if g16sub_return_code != 0:
                print(f"    g16sub failed (return code: {g16sub_return_code})")
                if stderr.strip():
                    print(f"    g16sub stderr: {stderr.strip()}")
                return None
            
            # 計算完了の監視（.outファイルの出現を待つ）
            out_file = self.monitor_job_completion(gjf_path)
            
            if out_file is None:
                print(f"    計算の完了を検出できませんでした: {base_name}")
                return None
            
            # 計算成功の判定（Normal terminationの確認）
            is_successful = self.check_calculation_success(out_file)
            
            if is_successful:
                return str(out_file)
            else:
                print(f"    計算が正常終了しませんでした: {base_name}")
                return None
                
        except Exception as e:
            print(f"    run_gaussian例外: {e}")
            return None
    
    def _nproc(self, job_suffix):
        """この状態のコア数"""
        return int(self.nproc_per_state.get(job_suffix, self.nproc))
    
    def _run_g16_direct(self, gjf_path, job_suffix=None):
        """ジョブ内でg16を直接実行し、終了を待つ（g16 < xxx.gjf > xxx.out）"""
        out_file = gjf_path.parent / f"{gjf_path.stem}.out"
        env = os.environ.copy()
        if job_suffix in self.cpu_sets:
            # 環境変数側も、この計算に割り当てたCPUにそろえる（gjfの%cpuと同じ）
            env['GAUSS_CDEF'] = self.cpu_sets[job_suffix]
        try:
            with open(gjf_path, 'r') as fin, open(out_file, 'w') as fout:
                subprocess.run(['g16'], stdin=fin, stdout=fout, stderr=subprocess.STDOUT,
                               cwd=gjf_path.parent, env=env)
        except Exception as e:
            print(f"    g16の実行に失敗しました: {e}")
            return None
        
        if self.check_calculation_success(str(out_file)):
            return str(out_file)
        print(f"    計算が正常終了しませんでした: {gjf_path.stem}")
        return None
    
    def calculate_single_state(self, charge, multiplicity, job_suffix):
        """単一状態の計算を実行
        
        Returns:
            tuple: (energy, job_name, job_suffix, output_file)
                   子プロセスからの戻り値として、親プロセスでoptimization_stateを更新するのに使う
        """
        job_name = FileNameGenerator.generate_job_name(
            self.base_name, job_suffix, self.omega, self.alpha, self.omega_digits
        )

        input_file = self.create_gaussian_input(charge, multiplicity, job_name, job_suffix)
        output_file = self.run_gaussian(input_file, job_suffix)

        if output_file is None:
            return (None, job_name, job_suffix, None)

        energy = GaussianResultParser.extract_energy(output_file)

        return (energy, job_name, job_suffix, output_file)
    
    def calculate_delta_scf(self, parallel=True, max_workers=3):
        """Delta SCF法によるIPとEAの計算
        
        q0, q1, qm1は独立した電子状態なので並列実行可能。
        各状態の計算完了後、親プロセス側でoptimization_stateを更新し、
        次の試行（ω変更時）でGuess=Read/oldchkが正しく機能するようにする。
        """
        
        # 計算する状態の定義
        states = [
            (self.charge, self.multiplicity, "q0"),
            (self.charge + 1, self.multiplicity + 1 if self.multiplicity == 1 else self.multiplicity - 1, "q1"),
            (self.charge - 1, self.multiplicity + 1 if self.multiplicity == 1 else self.multiplicity - 1, "qm1")
        ]
        
        if parallel and max_workers > 1:
            # 並列実行（子プロセスでの変更は親に返らないため、結果は戻り値で受け取る）
            state_results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.calculate_single_state, charge, mult, suffix): suffix
                           for charge, mult, suffix in states}
                for future in as_completed(futures):
                    try:
                        state_results.append(future.result())
                    except Exception as e:
                        print(f"Error in {futures[future]} calculation: {e}")
        else:
            # 逐次実行
            state_results = [self.calculate_single_state(charge, mult, suffix)
                             for charge, mult, suffix in states]

        energies = {}
        out_files = {}
        for energy, job_name, job_suffix, output_file in state_results:
            if energy is None:
                continue
            energies[job_suffix] = energy
            out_files[job_suffix] = output_file
            # 次の試行のGuess=Read用に、この状態の成功したchk名を親プロセス側で記録
            self.optimization_state.setdefault('previous_chk', {})[job_suffix] = job_name

        # 必要な全エネルギーが得られているかチェック
        if not all(key in energies for key in ['q0', 'q1', 'qm1']):
            return None

        # 中性分子とアニオンのHOMOを取得
        homo_energy_q0, lumo_energy_q0 = GaussianResultParser.extract_homo_lumo(out_files['q0'])
        homo_energy_qm1, lumo_energy_qm1 = GaussianResultParser.extract_homo_lumo(out_files['qm1'])
        if homo_energy_q0 is None or homo_energy_qm1 is None:
            return None
        
        # J値の計算
        j_results = JValueCalculator.calculate_j_value(energies, homo_energy_q0, homo_energy_qm1)
        
        results = {
            'E_q0': energies['q0'],
            'E_q1': energies['q1'],
            'E_qm1': energies['qm1'],
            'E_homo_q0': homo_energy_q0,
            'E_lumo_q0': lumo_energy_q0,
            'E_homo_qm1': homo_energy_qm1,
            'E_lumo_qm1': lumo_energy_qm1,
            'IP_hartree': (energies['q1'] - energies['q0']),
            'IP_eV': j_results['ip_ev'],
            'EA_hartree': (energies['q0'] - energies['qm1']),
            'EA_eV': j_results['ea_ev'],
            'E_homo_q0_eV': j_results['homo_ev_q0'],
            'E_homo_qm1_eV': j_results['homo_ev_qm1'],
            'J': j_results['j_value'],
            'omega': self.omega,
            'alpha': self.alpha
        }
        
        return results


class OmegaOptimizer:
    def __init__(self, config_file=None, **kwargs):
        """
        パラメーター最適化器（3つの最適化モード対応）
        """
        if config_file:
            self.config = self.load_config(config_file)
        else:
            self.config = self.create_default_config()
        
        # kwargsで設定を上書き
        for key, value in kwargs.items():
            if '.' in key:
                keys = key.split('.')
                current = self.config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                self.config[key] = value
        
        # 基本設定の取得
        self.omega_digits = self.config.get('optimization', {}).get('omega_digits', 2)
        if self.omega_digits not in [2, 3, 4]:
            raise ValueError("omega_digits must be 2, 3, or 4")
        
        # β関連設定の読み込み
        self.beta_mode = self.config.get('optimization', {}).get('beta_mode', 'default')
        self.beta_constraint = self.config.get('optimization', {}).get('beta_constraint')
        self.fixed_beta = self.config.get('optimization', {}).get('fixed_beta')
        self.epsilon = self.config.get('optimization', {}).get('epsilon', 1.0)
        
        # beta_modeの検証
        valid_beta_modes = ['default', 'constraint', 'fixed']
        if self.beta_mode not in valid_beta_modes:
            raise ValueError(f"beta_mode must be one of {valid_beta_modes}")
        
        # εの検証（beta_constraint='solvent'の場合のみ）
        if self._is_solvent_constraint():
            if self.epsilon < 1.0:
                raise ValueError("beta_constraint='solvent'の場合、ε（比誘電率）は1以上の正数である必要があります")
        
        # 最適化モードの取得と検証
        self.optimization_mode = self.config.get('optimization', {}).get('optimization_mode', 'omega_only')
        valid_modes = ['omega_only', 'alpha_only', 'omega_alpha']
        if self.optimization_mode not in valid_modes:
            raise ValueError(f"optimization_mode must be one of {valid_modes}")
        
        # 最適化アルゴリズムの設定
        default_algorithm = 'bayesian' if self.optimization_mode == 'omega_alpha' else 'brent'
        self.algorithm = self.config.get('optimization', {}).get('optimization_algorithm') or default_algorithm
        
        # アルゴリズムの検証
        valid_algorithms = ['bayesian', 'brent', 'golden_section', 'grid', 'adaptive_grid']
        if self.algorithm not in valid_algorithms:
            raise ValueError(f"optimization_algorithm must be one of {valid_algorithms}")
        
        # 1次元最適化でのみbrent, golden_sectionが使用可能
        if self.algorithm in ['brent', 'golden_section'] and self.optimization_mode not in ['omega_only', 'alpha_only']:
            raise ValueError(f"{self.algorithm} is only available for 1D optimization (omega_only or alpha_only)")
        
        print(f"\n選択された最適化アルゴリズム: {self.algorithm}")
        
        # gjfファイルからIOPパラメータを抽出
        self.gjf_omega, self.gjf_alpha = self.extract_gjf_parameters()
        
        # パラメータの決定（優先順位: yaml > gjf > default）
        self.omega_value, self.alpha_value = self.determine_parameters()
        
        # αの最大値の調整（beta_constraint='solvent'の場合のみε依存の制約を適用）
        original_alpha_max = self.config.get('optimization', {}).get('alpha_max')
        
        if self._is_solvent_constraint():
            epsilon_max = 1.0 / self.epsilon
            if original_alpha_max is None:
                self.alpha_max = epsilon_max
            else:
                self.alpha_max = min(original_alpha_max, epsilon_max)
        else:
            # beta_constraint='solvent'以外の場合は、εによる制約なし
            if original_alpha_max is None:
                self.alpha_max = 1.0  # デフォルト値
            else:
                self.alpha_max = original_alpha_max
        
        self.config['optimization']['alpha_max'] = self.alpha_max
        
        # 基本変数の初期化
        self.results_history = []
        self.best_omega = None
        self.best_alpha = None 
        self.best_j = float('inf')
        self.common_info_printed = False
        
        # パラメータ評価キャッシュ（同じ値の繰り返し評価を防止、再開時は既存結果も登録）
        # キー: (ω, α)（指定桁数で丸めた値、未指定はNone）
        self.evaluated_cache = {}
        self.existing_keys = set()  # 既存の.outから読み込んだ点（初回使用時に表示）
        
        # 最適化状態（各状態の前回成功したchk名。Guess=Read用）
        self.optimization_state = {'previous_chk': {}}
        
        # ジョブのwalltime（YAMLの calculation.walltime、形式を先に検証）
        self.walltime = self.config.get('calculation', {}).get('walltime', '2:00:00')
        walltime_to_seconds(self.walltime)
        
        # Gaussianの起動方法: 'g16sub'（1点ごとに投入）/ 'direct'（--submitで投入したジョブ内でg16を直接実行）
        self.launcher = self.config.get('calculation', {}).get('launcher', 'g16sub')
        self.mem_per_core_mb = int(self.config.get('calculation', {}).get('mem_per_core_mb', 1000))
        self.cpu_sets = None
        if self.launcher == 'direct':
            nprocs = state_nprocs(self.config)
            cdef = os.environ.get('GAUSS_CDEF')
            if cdef:
                self.cpu_sets = split_cpu_sets(cdef, nprocs, parallel=n_parallel_slots(self.config) > 1)
            if self.cpu_sets:
                print(f"ジョブ内実行: GAUSS_CDEF={cdef} を各状態に割り当てます")
                for state, cpus in self.cpu_sets.items():
                    print(f"  {state}: {nprocs[state]}コア  %cpu={cpus}")
            else:
                print(f"ジョブ内実行: GAUSS_CDEF={cdef} を分割できないため %nprocshared を使います")
        
        # chkファイルの自動削除（最適ωのchkは保持）
        # 計算ファイル（gjf/out/chk）の扱い: all（全部残す）/ optimal（最適点のout・chkだけ残す）/ none（全部消す）
        self.keep_files = self._get_opt_config('keep_files')
        if self.keep_files is None:
            # 旧オプション cleanup_chk: true は optimal として扱う
            self.keep_files = 'optimal' if self._get_opt_config('cleanup_chk', False) else 'all'
        self.keep_files = str(self.keep_files).lower()
        if self.keep_files not in ('all', 'optimal', 'none'):
            raise ValueError("keep_files must be one of 'all', 'optimal', 'none'")
        self.cleanup_chk = self.keep_files != 'all'  # 計算中に不要になったchkを削除するか
        self.created_chk = set()  # この実行で作成したchkのジョブ名
        
        # シグナルハンドラーの設定
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)

        # 既存結果の読み込み処理
        self.loaded_results = []
        
        previous_file = self.config.get('optimization', {}).get('previous_results_file')
        use_existing_outputs = self.config.get('optimization', {}).get('use_existing_outputs', False)
        
        if previous_file and use_existing_outputs:
            print("エラー: 'previous_results_file' と 'use_existing_outputs' の両方を同時に指定することはできません")
            sys.exit(1)
        
        if previous_file:
            self.load_previous_results(previous_file)
            print("既存のtxtファイルから結果を読み込みました")
        elif use_existing_outputs:
            existing_results = self.scan_existing_outputs()
            if existing_results:
                self.loaded_results = existing_results
                self.results_history.extend(existing_results)
                
                best_existing = min(existing_results, key=lambda x: x['j_value'])
                if best_existing['j_value'] < self.best_j:
                    self.best_j = best_existing['j_value']
                    self.best_omega = best_existing['omega']
                    self.best_alpha = best_existing['alpha']
                
                # 計算済みの点をキャッシュに登録（同じ点は再計算せずに再開できる）
                for r in existing_results:
                    key = self._cache_key(r['omega'], r['alpha'])
                    self.evaluated_cache[key] = r['j_value']
                    self.existing_keys.add(key)
                
                print("既存の.outファイルから結果を読み込みました")
            else:
                print("互換性のある既存の.outファイルが見つかりませんでした")
    
    def extract_gjf_parameters(self):
        """gjfファイルからIOPパラメータを抽出"""
        try:
            with open(self.config['molecule']['gjf_file'], 'r') as f:
                lines = f.readlines()
            return IOPParser.extract_iop_parameters(read_route_line(lines))
            
        except Exception as e:
            print(f"gjfファイルからのIOPパラメータ抽出に失敗: {e}")
            return None, None
    
    def determine_parameters(self):
        """パラメータの決定（優先順位: yaml > gjf > Gaussianのデフォルト）"""
        
        # yamlからの設定取得
        yaml_fixed_omega = self.config.get('optimization', {}).get('fixed_omega')
        yaml_fixed_alpha = self.config.get('optimization', {}).get('fixed_alpha')
        
        # ωの決定
        if self.optimization_mode == 'alpha_only':
            # α最適化：ωは固定
            if yaml_fixed_omega is not None:
                omega_value = yaml_fixed_omega
                print(f"ω固定値: {omega_value} (yaml設定)")
            elif self.gjf_omega is not None:
                omega_value = self.gjf_omega
                print(f"ω固定値: {omega_value} (gjfファイルから)")
            else:
                omega_value = None
                print(f"ω固定値: Gaussianのデフォルト値を使用 (IOP指定なし)")
        else:
            # ω最適化またはω+α最適化：ωは変数
            omega_value = None
        
        # αの決定
        if self.optimization_mode == 'omega_only':
            # ω最適化：αは固定
            if yaml_fixed_alpha is not None:
                alpha_value = yaml_fixed_alpha
                print(f"α固定値: {alpha_value} (yaml設定)")
            elif self.gjf_alpha is not None:
                alpha_value = self.gjf_alpha
                print(f"α固定値: {alpha_value} (gjfファイルから)")
            else:
                alpha_value = None
                print(f"α固定値: Gaussianのデフォルト値を使用 (IOP指定なし)")
        else:
            # α最適化またはω+α最適化：αは変数
            alpha_value = None
        
        return omega_value, alpha_value
    
    def _signal_handler(self, signum, frame):
        """Ctrl+C割り込みのハンドラー"""
        print("\n\n割り込みを検出しました。現在の結果を保存しています...")
        self.interrupted = True
        
        if self.results_history:
            if self.best_omega is not None:
                optimal_omega = self.best_omega
                optimal_j = self.best_j
                optimal_alpha = self.best_alpha
            else:
                best_result = min(self.results_history, key=lambda x: x['j_value'])
                optimal_omega = best_result['omega']
                optimal_j = best_result['j_value']
                optimal_alpha = best_result.get('alpha', None)
            
            self.save_results(optimal_omega, optimal_j, optimal_alpha)
            self.plot_results(optimal_omega, optimal_j, optimal_alpha)
            
            omega_format = f"{{:.{self.omega_digits}f}}"
            print(f"\n{'='*60}")
            print(f"最適化が中断されました")
            print(f"{'='*60}")
            print(f"中断時点での最良ω: {omega_format.format(optimal_omega)}")
            if optimal_alpha is not None:
                print(f"中断時点での最良α: {omega_format.format(optimal_alpha)}")
            print(f"中断時点での最小J値: {optimal_j:.6f} eV²")
            print(f"評価済み点数: {len(self.results_history)}")
            print(f"{'='*60}")
        else:
            print("まだ結果がないため、保存するデータがありません。")
        
        sys.exit(0)
    
    def scan_existing_outputs(self):
        """同フォルダ内の既存.outファイルをスキャンしてデータを抽出"""
        base_name = Path(self.config['molecule']['gjf_file']).stem
        current_dir = Path(self.config['molecule']['gjf_file']).parent
        
        try:
            temp_calculator = DeltaSCFCalculator(
                self.config['molecule']['gjf_file'], 
                nproc=1,
                omega_digits=self.omega_digits
            )
            current_method = temp_calculator.method.lower()
            current_basis = temp_calculator.basis.lower()
            current_charge = temp_calculator.charge
            current_mult = temp_calculator.multiplicity
        except Exception as e:
            print(f"現在の計算設定の読み取りに失敗: {e}")
            return []
        
        pattern = f"{base_name}_*.out"
        out_files = list(current_dir.glob(pattern))
        
        if not out_files:
            return []
        
        print(f"既存の.outファイルを検索中: {len(out_files)}個のファイルを発見")
        
        extracted_results = []
        compatible_files = 0
        
        for out_file in out_files:
            try:
                filename = out_file.stem
                omega, alpha = FileNameGenerator.extract_parameters_from_filename(filename, self.optimization_mode)
                
                if (self.optimization_mode == 'omega_only' and omega is None) or \
                   (self.optimization_mode == 'alpha_only' and alpha is None) or \
                   (self.optimization_mode == 'omega_alpha' and (omega is None or alpha is None)):
                    continue
                
                # 固定パラメータ（omega_onlyのα、alpha_onlyのω）とβ設定が現在の設定と一致するもののみ
                if not self._matches_current_settings(out_file, omega, alpha):
                    continue
                
                calc_info = GaussianResultParser.extract_calculation_info(out_file)
                if not calc_info:
                    continue
                
                if (calc_info['method'] != current_method or 
                    calc_info['basis'] != current_basis or 
                    calc_info['charge'] != current_charge or 
                    calc_info['multiplicity'] != current_mult):
                    continue
                
                state_results = self.extract_state_results(out_file, omega, alpha)
                if state_results:
                    extracted_results.append(state_results)
                    compatible_files += 1
                    
            except Exception as e:
                print(f"ファイル {out_file.name} の処理中にエラー: {e}")
                continue
        
        print(f"互換性のある計算結果: {compatible_files}セット")
        return extracted_results

    def extract_state_results(self, out_file, omega, alpha):
        """状態別の計算結果を抽出"""
        base_name = Path(self.config['molecule']['gjf_file']).stem
        current_dir = out_file.parent
        
        q0_pattern = FileNameGenerator.generate_job_name(
            base_name, "q0", omega, alpha, self.omega_digits
        ) + ".out"
        q1_pattern = FileNameGenerator.generate_job_name(
            base_name, "q1", omega, alpha, self.omega_digits
        ) + ".out"
        qm1_pattern = FileNameGenerator.generate_job_name(
            base_name, "qm1", omega, alpha, self.omega_digits
        ) + ".out"
        
        q0_file = current_dir / q0_pattern
        q1_file = current_dir / q1_pattern
        qm1_file = current_dir / qm1_pattern
        
        if not all([q0_file.exists(), q1_file.exists(), qm1_file.exists()]):
            return None
        
        try:
            energies = {}
            for state, file_path in [('q0', q0_file), ('q1', q1_file), ('qm1', qm1_file)]:
                energy = GaussianResultParser.extract_energy(file_path)
                if energy is None:
                    return None
                energies[state] = energy
            
            homo_energy_q0, lumo_energy_q0 = GaussianResultParser.extract_homo_lumo(q0_file)
            if homo_energy_q0 is None:
                return None
            
            homo_energy_qm1, lumo_energy_qm1 = GaussianResultParser.extract_homo_lumo(qm1_file)
            if homo_energy_qm1 is None:
                return None
            
            j_results = JValueCalculator.calculate_j_value(energies, homo_energy_q0, homo_energy_qm1)
            
            return {
                'omega': omega,
                'alpha': alpha,
                'j_value': j_results['j_value'],
                'ip_ev': j_results['ip_ev'],
                'ea_ev': j_results['ea_ev'],
                'homo_ev_q0': j_results['homo_ev_q0'],
                'homo_ev_qm1': j_results['homo_ev_qm1']
            }
            
        except Exception as e:
            print(f"結果抽出エラー (ω={omega}, α={alpha}): {e}")
            return None
    
    def load_previous_results(self, results_file):
        """以前の最適化結果ファイル（txt）を読み込む"""
        results_path = Path(results_file)
        
        if not results_path.exists():
            print(f"警告: 指定された結果ファイルが見つかりません: {results_file}")
            return
        
        print(f"既存の結果ファイルを読み込み中: {results_file}")
        
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
            
            # "Evaluation History:"のセクションを探す
            history_start = None
            for i, line in enumerate(lines):
                if "Evaluation History:" in line:
                    history_start = i + 2  # ヘッダーとセパレータをスキップ
                    break
            
            if history_start is None:
                print("警告: 評価履歴セクションが見つかりませんでした")
                return
            
            # ヘッダー行をスキップ（"Omega		Alpha		J Value..."）
            data_start = history_start + 1
            
            loaded_count = 0
            for line in lines[data_start:]:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                # タブ区切りで、空文字列を除去
                parts = [p.strip() for p in parts if p.strip()]
                
                if len(parts) < 7:
                    continue
                
                try:
                    omega_str = parts[0]
                    alpha_str = parts[1]
                    j_value = float(parts[2])
                    ip_ev = float(parts[3])
                    ea_ev = float(parts[4])
                    homo_ev_q0 = float(parts[5])
                    homo_ev_qm1 = float(parts[6])
                    
                    # "default"の場合はNoneに変換
                    omega = None if omega_str == "default" else float(omega_str)
                    alpha = None if alpha_str == "default" else float(alpha_str)
                    
                    result = {
                        'omega': omega,
                        'alpha': alpha,
                        'j_value': j_value,
                        'ip_ev': ip_ev,
                        'ea_ev': ea_ev,
                        'homo_ev_q0': homo_ev_q0,
                        'homo_ev_qm1': homo_ev_qm1
                    }
                    
                    self.loaded_results.append(result)
                    loaded_count += 1
                    
                except (ValueError, IndexError):
                    continue
            
            print(f"  {loaded_count}個の評価結果を読み込みました")
            
        except Exception as e:
            print(f"結果ファイルの読み込みエラー: {e}")
            return
    
    def load_config(self, config_file):
        """設定ファイルを読み込み"""
        config_path = Path(config_file)
        
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            raise ValueError("設定ファイルはYAMLまたはJSON形式である必要があります")
    
    def _get_opt_config(self, key, default=None):
        """optimization設定を簡潔に取得するヘルパー"""
        return self.config.get('optimization', {}).get(key, default)

    def _is_solvent_constraint(self):
        return (self.beta_mode == 'constraint' and isinstance(self.beta_constraint, str)
                and self.beta_constraint.lower() == 'solvent')

    def _beta_description(self):
        """βの取り扱いの説明文（表示用）"""
        if self.beta_mode == 'constraint':
            if self._is_solvent_constraint():
                return f"β制約: α + β = 1/ε = {1.0 / self.epsilon:.{self.omega_digits}f} (ε = {self.epsilon})"
            return f"β制約: α + β = {self.beta_constraint}"
        if self.beta_mode == 'fixed':
            return f"固定β値: {self.fixed_beta}"
        return "Gaussianデフォルト値を使用"

    def _beta_value(self, alpha):
        """αに対応するβ（IOP指定なしの場合はNone）"""
        if alpha is None or self.beta_mode == 'default':
            return None
        if self.beta_mode == 'fixed':
            return self.fixed_beta
        C = 1.0 / self.epsilon if self._is_solvent_constraint() else float(self.beta_constraint)
        return C - alpha
    
    def _cache_key(self, omega, alpha):
        """評価キャッシュのキー（指定桁数で丸めた(ω, α)）"""
        r = lambda v: round(v, self.omega_digits) if v is not None else None
        return (r(omega), r(alpha))
    
    def _job_names(self, omega, alpha):
        """(ω, α)の各状態のジョブ名（= gjf/out/chkのファイル名の幹）"""
        base_name = Path(self.config['molecule']['gjf_file']).stem
        omega, alpha = self._cache_key(omega, alpha)
        return {state: FileNameGenerator.generate_job_name(base_name, state, omega, alpha, self.omega_digits)
                for state in ('q0', 'q1', 'qm1')}
    
    def _matches_current_settings(self, out_file, omega, alpha):
        """既存の計算結果が現在の固定パラメータ・β設定と一致するか"""
        if self.optimization_mode == 'omega_only' and \
                self._cache_key(None, alpha) != self._cache_key(None, self.alpha_value):
            return False
        if self.optimization_mode == 'alpha_only' and \
                self._cache_key(omega, None) != self._cache_key(self.omega_value, None):
            return False
        
        # βはファイル名に含まれないため、同名のgjfのルート行で確認
        gjf_file = Path(out_file).with_suffix('.gjf')
        if not gjf_file.exists():
            return True
        with open(gjf_file, 'r') as f:
            route = read_route_line(f.readlines()).lower()
        beta = self._beta_value(alpha)
        if beta is None:
            return '3/119=' not in route
        beta_str = f"{round(round(beta, self.omega_digits) * 10000):05d}00000"
        return f"3/119={beta_str}" in route
    
    def _register_existing_chk(self, omega, alpha):
        """既存の計算点を再利用したとき、そのchkを次の計算のGuess=Read用に登録"""
        for state, name in self._job_names(omega, alpha).items():
            if os.path.exists(f"{name}.chk"):
                self.optimization_state['previous_chk'][state] = name
    
    def _cleanup_chk(self):
        """計算中に、この実行で作成したchkのうち不要になったものを削除（keep_files: optimal / none）
        
        各状態の最新chk（次のGuess=Read用）と現時点の最良点のchkは保持する
        """
        if not self.cleanup_chk:
            return 0
        
        keep = set(self.optimization_state['previous_chk'].values())
        if self.best_j < float('inf'):
            keep |= set(self._job_names(self.best_omega, self.best_alpha).values())
        
        deleted = 0
        for name in sorted(self.created_chk - keep):
            try:
                os.remove(f"{name}.chk")
                deleted += 1
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"    chkファイルを削除できませんでした: {name}.chk ({e})")
                continue
            self.created_chk.discard(name)
        return deleted
    
    def _cleanup_files(self, optimal_omega, optimal_alpha):
        """最適化の完了後に計算ファイルを整理する（keep_files に従う）
        
        対象は、この最適化で評価・再利用したすべての点（再開時に読み込んだ点を含む）の
        .gjf / .gjf.ap / .out / .out.old / .chk。結果ファイル・グラフ・ログは残す。
        中断時（Ctrl+C、walltime超過）はここを通らないので、再開に必要なファイルは消えない。
        """
        if self.keep_files == 'all':
            return
        
        keep = set()
        if self.keep_files == 'optimal':
            # 報告した最適点が評価済みでなければ、評価済みの最良点を残す
            key = self._cache_key(optimal_omega, optimal_alpha)
            if key not in self.evaluated_cache and self.best_j < float('inf'):
                key = self._cache_key(self.best_omega, self.best_alpha)
            keep = set(self._job_names(*key).values())
        
        names = set()
        for omega, alpha in self.evaluated_cache:
            names |= set(self._job_names(omega, alpha).values())
        
        deleted = 0
        for name in sorted(names):
            extensions = ['.gjf', '.gjf.ap', '.out.old']
            if name not in keep:
                extensions += ['.out', '.chk']
            for ext in extensions:
                try:
                    os.remove(f"{name}{ext}")
                    deleted += 1
                except FileNotFoundError:
                    pass
                except OSError as e:
                    print(f"    削除できませんでした: {name}{ext} ({e})")
        
        if self.keep_files == 'optimal':
            print(f"計算ファイルを{deleted}個削除しました（最適点の .out と .chk は保持）")
        else:
            print(f"計算ファイルを{deleted}個削除しました（結果ファイルとグラフは保持）")
    
    def _get_1d_param_settings(self):
        """1次元最適化用のパラメータ設定を取得
        
        Returns:
            dict: param_min, param_max, param_name, fixed_param, fixed_name を含む辞書
        
        Raises:
            ValueError: omega_alphaモードの場合
        """
        if self.optimization_mode == 'omega_only':
            return {
                'param_min': self._get_opt_config('omega_min'),
                'param_max': self._get_opt_config('omega_max'),
                'param_name': 'ω',
                'fixed_param': self.alpha_value,
                'fixed_name': 'α'
            }
        elif self.optimization_mode == 'alpha_only':
            return {
                'param_min': self._get_opt_config('alpha_min'),
                'param_max': self._get_opt_config('alpha_max'),
                'param_name': 'α',
                'fixed_param': self.omega_value,
                'fixed_name': 'ω'
            }
        else:
            raise ValueError("このメソッドは1次元最適化（omega_only, alpha_only）専用です")
    
    def _set_optimal_params_by_mode(self, optimal_param):
        """最適化モードに応じてoptimal_omega, optimal_alphaを設定
        
        Args:
            optimal_param: 最適化されたパラメータ値
            
        Returns:
            tuple: (optimal_omega, optimal_alpha)
        """
        if self.optimization_mode == 'omega_only':
            return optimal_param, self.alpha_value
        elif self.optimization_mode == 'alpha_only':
            return self.omega_value, optimal_param
        else:
            # omega_alphaの場合はこのメソッドを使わない
            raise ValueError("omega_alphaモードではこのメソッドは使用できません")
    
    def _finalize_optimization(self, optimal_omega, optimal_j, optimal_alpha, method_name, extra_info=None):
        """最適化の終了処理（結果保存、プロット、結果表示）
        
        Args:
            optimal_omega: 最適ω値
            optimal_j: 最適J値
            optimal_alpha: 最適α値
            method_name: 最適化手法名（表示用）
            extra_info: 追加で表示する情報のdict（キー: 表示名, 値: 値）
        """
        self.save_results(optimal_omega, optimal_j, optimal_alpha)
        self.plot_results(optimal_omega, optimal_j, optimal_alpha)
        
        # 計算ファイルの整理（keep_files: optimal / none のとき）
        self._cleanup_files(optimal_omega, optimal_alpha)
        
        print(f"\n{'='*60}")
        print(f"{method_name}完了")
        print(f"{'='*60}")
        
        fmt = lambda v: f"{v:.{self.omega_digits}f}" if v is not None else "Gaussianデフォルト"
        if self.optimization_mode == 'omega_only':
            print(f"最適ω: {fmt(optimal_omega)}")
            print(f"固定α: {fmt(optimal_alpha)}")
        elif self.optimization_mode == 'alpha_only':
            print(f"固定ω: {fmt(optimal_omega)}")
            print(f"最適α: {fmt(optimal_alpha)}")
        elif self.optimization_mode == 'omega_alpha':
            print(f"最適ω: {fmt(optimal_omega)}")
            print(f"最適α: {fmt(optimal_alpha)}")
        
        beta_value = self._beta_value(optimal_alpha)
        if beta_value is not None:
            print(f"β: {beta_value:.{self.omega_digits}f}")
        
        print(f"最小J値: {optimal_j:.6f} eV²")
        
        if extra_info:
            for name, value in extra_info.items():
                print(f"{name}: {value}")
        
        print(f"総評価回数: {len(self.results_history)}")
        print(f"{'='*60}")
    
    def _print_1d_optimization_header(self, method_name, params, extra_info=None):
        """1次元最適化のヘッダー表示
        
        Args:
            method_name: 最適化手法名
            params: _get_1d_param_settings()の戻り値
            extra_info: 追加情報のdict
        """
        print(f"{method_name}による{params['param_name']}最適化:")
        print(f"  {params['param_name']}範囲: [{params['param_min']:.{self.omega_digits}f}, {params['param_max']:.{self.omega_digits}f}]")
        if params['fixed_param'] is not None:
            print(f"  {params['fixed_name']}固定値: {params['fixed_param']:.{self.omega_digits}f}")
        else:
            print(f"  {params['fixed_name']}固定値: Gaussianデフォルト")
        
        if extra_info:
            for name, value in extra_info.items():
                print(f"  {name}: {value}")
    
    def create_default_config(self):
        """デフォルト設定を作成"""
        return default_config()
    
    def print_common_info(self):
        """共通情報を一度だけ表示"""
        if not self.common_info_printed:
            print("="*60)
            print("3つの最適化モード対応版での最適化を開始します...")
            print("="*60)
            
            temp_calculator = DeltaSCFCalculator(
                self.config['molecule']['gjf_file'], 
                nproc=1,
                omega_digits=self.omega_digits
            )
            
            print(f"分子: {self.config['molecule']['gjf_file']}")
            print(f"汎関数: {temp_calculator.method}")
            print(f"基底関数: {temp_calculator.basis}")
            print(f"電荷: {temp_calculator.charge}, 多重度: {temp_calculator.multiplicity}")
            nprocs = state_nprocs(self.config)
            print("コア数: " + ", ".join(f"{s} {nprocs[s]}" for s in STATES))
            
            print(f"最適化モード: {self.optimization_mode}")
            
            if self.optimization_mode == 'omega_only':
                print(f"ω範囲: {self.config['optimization']['omega_min']:.{self.omega_digits}f} - " +
                      f"{self.config['optimization']['omega_max']:.{self.omega_digits}f}")
                if self.alpha_value is not None:
                    print(f"α固定値: {self.alpha_value:.{self.omega_digits}f}")
                else:
                    print(f"α固定値: Gaussianのデフォルト値を使用")
            elif self.optimization_mode == 'alpha_only':
                print(f"α範囲: {self.config['optimization']['alpha_min']:.{self.omega_digits}f} - " +
                      f"{self.alpha_max:.{self.omega_digits}f}")
                if self.omega_value is not None:
                    print(f"ω固定値: {self.omega_value:.{self.omega_digits}f}")
                else:
                    print(f"ω固定値: Gaussianのデフォルト値を使用")
            elif self.optimization_mode == 'omega_alpha':
                print(f"ω範囲: {self.config['optimization']['omega_min']:.{self.omega_digits}f} - " +
                      f"{self.config['optimization']['omega_max']:.{self.omega_digits}f}")
                print(f"α範囲: {self.config['optimization']['alpha_min']:.{self.omega_digits}f} - " +
                      f"{self.alpha_max:.{self.omega_digits}f}")
            
            print(f"桁数: 小数点以下{self.omega_digits}桁")
            print(f"最大評価回数: {self.config['optimization']['n_calls']}")
            
            use_guess = self.config.get('optimization', {}).get('use_guess_read', True)
            print(f"Guess=Read (前回chk再利用): {'有効' if use_guess else '無効'}")
            if self.launcher == 'direct':
                print("Gaussianの実行: このジョブ内でg16を直接実行")
            else:
                print(f"Gaussianの実行: 1点ごとにg16subで投入（walltime: {seconds_to_walltime(walltime_to_seconds(self.walltime))}）")
            keep_label = {'all': '全部残す', 'optimal': '最適点の .out と .chk だけ残す',
                          'none': '全部消す（結果ファイルとグラフは残す）'}[self.keep_files]
            print(f"計算ファイル（完了後）: {keep_label}")
            
            print("="*60)
            print()
            
            self.common_info_printed = True
    
    def objective_function(self, param):
        """目的関数：指定されたパラメータでJ値を計算
        
        計算済みの点（この実行中、または再開時に読み込んだ既存の.out）は再計算せずに
        キャッシュの値を返す。
        """
        if self.interrupted:
            raise KeyboardInterrupt("Optimization interrupted by user")
            
        self.print_common_info()
        
        fmt = lambda v: f"{v:.{self.omega_digits}f}" if v is not None else "Gaussianデフォルト"
        
        # パラメータの処理
        if self.optimization_mode == 'omega_only':
            omega = round(param[0], self.omega_digits)
            alpha = self.alpha_value  # 固定値
            label = f"ω = {fmt(omega)} (最適化), α = {fmt(alpha)} (固定)"
        elif self.optimization_mode == 'alpha_only':
            alpha = round(param[0], self.omega_digits)
            omega = self.omega_value  # 固定値
            label = f"ω = {fmt(omega)} (固定), α = {fmt(alpha)} (最適化)"
        else:  # omega_alpha
            omega = round(param[0], self.omega_digits)
            alpha = round(param[1], self.omega_digits)
            label = f"ω = {fmt(omega)}, α = {fmt(alpha)}"
        
        key = self._cache_key(omega, alpha)
        if key in self.evaluated_cache:
            j_value = self.evaluated_cache[key]
            if key in self.existing_keys:
                # 再開時：既存の計算結果を使用（初回のみ表示）し、そのchkを次のGuess=Readに使う
                self.existing_keys.discard(key)
                print(f"\n{label}: 既存の計算結果を使用 (J = {j_value:.6f} eV²)")
                self._register_existing_chk(omega, alpha)
            self._check_j_threshold(j_value)
            return j_value
        
        print(f"\n{label} での計算開始")
        
        calculator = DeltaSCFCalculator(
            self.config['molecule']['gjf_file'],
            nproc=self.config['calculation']['nproc_per_job'],
            nproc_per_state=state_nprocs(self.config),
            omega=omega,
            omega_digits=self.omega_digits,
            alpha=alpha,
            epsilon=self.epsilon,
            beta_mode=self.beta_mode,
            beta_constraint=self.beta_constraint,
            fixed_beta=self.fixed_beta,
            optimization_state=self.optimization_state,
            use_guess_read=self.config['optimization'].get('use_guess_read', True),
            walltime=self.walltime,
            launcher=self.launcher,
            cpu_sets=self.cpu_sets,
            mem_per_core_mb=self.mem_per_core_mb
        )
        
        # 並列実行の設定
        max_parallel = self.config['optimization'].get('max_parallel_jobs', 1)
        parallel = max_parallel > 1
        
        results = calculator.calculate_delta_scf(
            parallel=parallel,
            max_workers=min(max_parallel, 3)
        )
        self.created_chk.update(self._job_names(omega, alpha).values())
        
        if results is None:
            print(f"計算に失敗しました")
            self.evaluated_cache[key] = 1000.0
            self._cleanup_chk()
            return 1000.0
        
        j_value = results['J']
        self.evaluated_cache[key] = j_value
        
        # 結果を記録
        result_record = {
            'omega': omega,
            'alpha': alpha,
            'j_value': j_value,
            'ip_ev': results['IP_eV'],
            'ea_ev': results['EA_eV'],
            'homo_ev_q0': results['E_homo_q0_eV'],
            'homo_ev_qm1': results['E_homo_qm1_eV']
        }
        
        self.results_history.append(result_record)
        
        # 最良結果の更新
        if j_value < self.best_j:
            self.best_j = j_value
            self.best_omega = omega
            self.best_alpha = alpha
        
        # 不要になったchkの削除（cleanup_chk有効時のみ）
        self._cleanup_chk()
        
        print(f"  J値: {j_value:.6f} eV²")
        print(f"  IP: {results['IP_eV']:.3f} eV, EA: {results['EA_eV']:.3f} eV")
        print(f"  E(HOMO_q0): {results['E_homo_q0_eV']:.3f} eV, E(HOMO_qm1): {results['E_homo_qm1_eV']:.3f} eV")
        
        # 最良結果の表示
        if self.optimization_mode == 'omega_only':
            best_label = f"ω = {fmt(self.best_omega)}"
        elif self.optimization_mode == 'alpha_only':
            best_label = f"α = {fmt(self.best_alpha)}"
        elif self.best_alpha is not None:
            best_label = f"ω = {fmt(self.best_omega)}, α = {fmt(self.best_alpha)}"
        else:
            best_label = f"ω = {fmt(self.best_omega)}"
        print(f"  現在の最良: {best_label}, J = {self.best_j:.6f} eV²")
        
        self._check_j_threshold(j_value)
        return j_value
    
    def _check_j_threshold(self, j_value):
        """J値が閾値以下なら早期終了（StopIteration）"""
        j_threshold = self.config['optimization'].get('j_threshold')
        if j_threshold is not None and j_value <= j_threshold:
            print(f"\nJ値 ({j_value:.6f}) が閾値 ({j_threshold}) 以下になりました")
            raise StopIteration()
    
    def optimize(self):
        """選択されたアルゴリズムで最適化を実行"""
        print(f"\n{'='*60}")
        print(f"最適化アルゴリズム: {self.algorithm}")
        print(f"最適化モード: {self.optimization_mode}")
        
        # βパラメータの取り扱いを表示
        print(f"βパラメータの取り扱い: {self.beta_mode}")
        print(f"  {self._beta_description()}")

        print(f"{'='*60}\n")
        
        if self.algorithm == 'bayesian':
            return self.optimize_bayesian()
        elif self.algorithm == 'brent':
            return self.optimize_brent()
        elif self.algorithm == 'golden_section':
            return self.optimize_golden_section()
        elif self.algorithm == 'grid':
            return self.optimize_grid()
        elif self.algorithm == 'adaptive_grid':
            return self.optimize_adaptive_grid()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def optimize_bayesian(self):
        """ベイズ最適化（既存の実装）"""
        if not SKOPT_AVAILABLE:
            print("エラー: scikit-optimizeが必要です")
            print("pip install scikit-optimize でインストールしてください")
            sys.exit(1)
        
        # 最適化空間の定義
        if self.optimization_mode == 'omega_only':
            space = [Real(
                self.config['optimization']['omega_min'],
                self.config['optimization']['omega_max'],
                name='omega'
            )]
        elif self.optimization_mode == 'alpha_only':
            space = [Real(
                self.config['optimization']['alpha_min'],
                self.alpha_max,
                name='alpha'
            )]
        elif self.optimization_mode == 'omega_alpha':
            space = [
                Real(self.config['optimization']['omega_min'],
                     self.config['optimization']['omega_max'],
                     name='omega'),
                Real(self.config['optimization']['alpha_min'],
                     self.alpha_max,
                     name='alpha')
            ]
        
        # 既存データの準備
        x0, y0 = self.prepare_initial_data_for_bayes()
        
        new_calls = self.config['optimization']['n_calls']
        if x0 is not None and len(x0) > 0:
            existing_count = len(x0)
            print(f"既存データ: {existing_count}点, 新規計算予定: {new_calls}点")
        else:
            print(f"新規計算予定: {new_calls}点")
        
        # ベイズ最適化の設定
        n_initial = self.config['optimization'].get('n_initial_points', 8)
        acq_func = self.config['optimization'].get('bayesian_acq_func', 'gp_hedge')
        
        # 1次元と2次元で推奨値を調整
        if self.optimization_mode in ['omega_only', 'alpha_only']:
            if n_initial < 8:
                n_initial = 8
                print(f"  警告: 1次元最適化では初期点を{n_initial}点に設定することを推奨")
        elif self.optimization_mode == 'omega_alpha':
            if n_initial < 15:
                print(f"  警告: 2次元最適化では初期点を15点以上に設定することを推奨")
        
        print(f"ベイズ最適化設定:")
        print(f"  評価回数: {new_calls}")
        print(f"  初期点数: {n_initial}")
        print(f"  獲得関数: {acq_func}")
        
        try:
            def wrapped_objective(param):
                try:
                    return self.objective_function(param)
                except StopIteration:
                    raise ValueError("Early termination due to J threshold")
            
            gp_minimize(
                func=wrapped_objective,
                dimensions=space,
                n_calls=new_calls,
                n_initial_points=n_initial,
                acq_func=acq_func,
                initial_point_generator='lhs',
                random_state=42,
                x0=x0,
                y0=y0
            )            
            
            optimal_omega = self.best_omega
            optimal_alpha = self.best_alpha
            optimal_j = self.best_j
            
        except ValueError as e:
            if "Early termination" in str(e):
                print(f"最適化が早期終了しました")
                optimal_omega = self.best_omega
                optimal_alpha = self.best_alpha
                optimal_j = self.best_j
            else:
                raise e
        except Exception as e:
            print(f"最適化中にエラーが発生しました: {e}")
            if self.best_omega is not None:
                optimal_omega = self.best_omega
                optimal_alpha = self.best_alpha
                optimal_j = self.best_j
                print(f"これまでの最良結果を使用します")
            else:
                raise e
        
        self._finalize_optimization(optimal_omega, optimal_j, optimal_alpha, 'ベイズ最適化')
        
        return optimal_omega, optimal_j
    
    def prepare_initial_data_for_bayes(self):
        """ベイズ最適化用の初期データ(x0, y0)を準備"""
        if not self.loaded_results:
            return None, None
        
        x0 = []
        y0 = []
        
        for result in self.loaded_results:
            omega_rounded = round(result['omega'], self.omega_digits)
            alpha_rounded = round(result['alpha'], self.omega_digits) if result['alpha'] is not None else 0.0
            
            if self.optimization_mode == 'omega_only':
                x0.append([omega_rounded])
            elif self.optimization_mode == 'alpha_only':
                x0.append([alpha_rounded])
            elif self.optimization_mode == 'omega_alpha':
                x0.append([omega_rounded, alpha_rounded])
            
            y0.append(result['j_value'])
        
        return x0, y0
    
    def optimize_brent(self):
        """Brent法による1次元最適化（単峰性関数に最適）"""
        if not SCIPY_AVAILABLE:
            print("エラー: scipyが必要です")
            print("pip install scipy でインストールしてください")
            sys.exit(1)
        
        # パラメータ設定を取得
        params = self._get_1d_param_settings()
        max_iter = self._get_opt_config('n_calls')
        tolerance = 10 ** (-self.omega_digits - 1)
        
        # ヘッダー表示
        self._print_1d_optimization_header('Brent法', params, {
            '最大反復': max_iter,
            '許容誤差': tolerance
        })
        
        # 連続同一値判定用の変数
        consecutive_same_count = 0
        last_evaluated_param = None
        convergence_threshold = 5
        
        def objective_wrapper(param):
            nonlocal consecutive_same_count, last_evaluated_param
            param_rounded = round(param, self.omega_digits)
            
            if param_rounded == last_evaluated_param:
                consecutive_same_count += 1
                if consecutive_same_count >= convergence_threshold:
                    print(f"\n収束判定: 連続{consecutive_same_count}回同じ値({param_rounded:.{self.omega_digits}f})が選択されました")
                    raise ValueError("Convergence detected: same parameter value repeated")
            else:
                consecutive_same_count = 0
                last_evaluated_param = param_rounded
            
            try:
                # 計算済みの点はobjective_function内のキャッシュで処理される
                return self.objective_function([param_rounded])
            except StopIteration:
                raise ValueError("Early termination due to J threshold")
        
        extra_info = {}
        
        try:
            result = minimize_scalar(
                objective_wrapper,
                bounds=(params['param_min'], params['param_max']),
                method='bounded',
                options={'maxiter': max_iter, 'xatol': tolerance}
            )
            optimal_param = round(result.x, self.omega_digits)
            optimal_j = result.fun
            
            if hasattr(result, 'nit'):
                extra_info['反復回数'] = result.nit
            if hasattr(result, 'nfev'):
                extra_info['関数評価回数'] = result.nfev
            
        except ValueError as e:
            error_msg = str(e)
            if "Early termination" in error_msg:
                print(f"\n最適化が早期終了しました（J値が閾値以下）")
            elif "Convergence detected" in error_msg:
                print(f"\n最適化が収束しました（これ以上の探索点なし）")
            else:
                raise e
            optimal_param = self.best_omega if self.optimization_mode == 'omega_only' else self.best_alpha
            optimal_j = self.best_j
        
        # 結果の設定
        optimal_omega, optimal_alpha = self._set_optimal_params_by_mode(optimal_param)
        
        # 終了処理
        self._finalize_optimization(optimal_omega, optimal_j, optimal_alpha, 'Brent法による最適化', extra_info)
        
        return optimal_omega, optimal_j
        
    def optimize_golden_section(self):
        """黄金分割探索による1次元最適化"""
        PHI = (1 + np.sqrt(5)) / 2
        
        # パラメータ設定を取得
        params = self._get_1d_param_settings()
        a, b = params['param_min'], params['param_max']
        tolerance = 10 ** (-self.omega_digits - 1)
        max_iterations = self._get_opt_config('n_calls')
        
        # ヘッダー表示
        self._print_1d_optimization_header('黄金分割探索', params, {
            '最大反復': max_iterations,
            '許容誤差': tolerance
        })
        
        # 初期の内分点
        x1 = b - (b - a) / PHI
        x2 = a + (b - a) / PHI
        iteration = 0
        
        # 初期評価
        try:
            f1 = self.objective_function([round(x1, self.omega_digits)])
            f2 = self.objective_function([round(x2, self.omega_digits)])
            iteration = 2
        except StopIteration:
            optimal_omega, optimal_alpha = self._set_optimal_params_by_mode(
                self.best_omega if self.optimization_mode == 'omega_only' else self.best_alpha
            )
            self._finalize_optimization(optimal_omega, self.best_j, optimal_alpha, '黄金分割探索（早期終了）')
            return optimal_omega, self.best_j
        
        # 探索ループ
        try:
            while abs(b - a) > tolerance and iteration < max_iterations:
                print(f"\n反復 {iteration}: 探索範囲 [{a:.{self.omega_digits}f}, {b:.{self.omega_digits}f}]")
                
                if f1 < f2:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = b - (b - a) / PHI
                    f1 = self.objective_function([round(x1, self.omega_digits)])
                else:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = a + (b - a) / PHI
                    f2 = self.objective_function([round(x2, self.omega_digits)])
                
                iteration += 1
                current_best = round(x1 if f1 < f2 else x2, self.omega_digits)
                current_j = min(f1, f2)
                print(f"  現在の最良: {params['param_name']} = {current_best:.{self.omega_digits}f}, J = {current_j:.6f}")
        
        except StopIteration:
            print(f"\n最適化が早期終了しました（J値が閾値以下）")
        
        # 結果の設定と終了処理
        optimal_param = self.best_omega if self.optimization_mode == 'omega_only' else self.best_alpha
        optimal_omega, optimal_alpha = self._set_optimal_params_by_mode(optimal_param)
        
        self._finalize_optimization(optimal_omega, self.best_j, optimal_alpha, '黄金分割探索', 
                                    {'総反復回数': iteration})
        
        return optimal_omega, self.best_j
        
    def optimize_grid(self):
        """グリッドサーチ"""
        n_points = self._get_opt_config('n_calls')
        
        print(f"グリッドサーチ:")
        
        if self.optimization_mode in ['omega_only', 'alpha_only']:
            params = self._get_1d_param_settings()
            param_values = np.linspace(params['param_min'], params['param_max'], n_points)
            
            print(f"  {params['param_name']}方向に{n_points}点を等間隔評価")
            if params['fixed_param'] is not None:
                print(f"  {params['fixed_name']}固定値: {params['fixed_param']:.{self.omega_digits}f}")
            
            try:
                for i, val in enumerate(param_values, 1):
                    val_rounded = round(val, self.omega_digits)
                    print(f"\n[{i}/{n_points}] {params['param_name']} = {val_rounded:.{self.omega_digits}f}")
                    self.objective_function([val_rounded])
            except StopIteration:
                print(f"\n最適化が早期終了しました（J値が閾値以下）")
        
        elif self.optimization_mode == 'omega_alpha':
            n_omega = int(np.sqrt(n_points))
            n_alpha = n_points // n_omega
            
            omega_values = np.linspace(
                self._get_opt_config('omega_min'),
                self._get_opt_config('omega_max'),
                n_omega
            )
            alpha_values = np.linspace(
                self._get_opt_config('alpha_min'),
                self._get_opt_config('alpha_max'),
                n_alpha
            )
            
            print(f"  {n_omega} × {n_alpha} = {n_omega * n_alpha}点を評価")
            
            count = 0
            try:
                for omega in omega_values:
                    for alpha in alpha_values:
                        count += 1
                        omega_rounded = round(omega, self.omega_digits)
                        alpha_rounded = round(alpha, self.omega_digits)
                        print(f"\n[{count}/{n_omega * n_alpha}] ω = {omega_rounded:.{self.omega_digits}f}, "
                              f"α = {alpha_rounded:.{self.omega_digits}f}")
                        self.objective_function([omega_rounded, alpha_rounded])
            except StopIteration:
                print(f"\n最適化が早期終了しました（J値が閾値以下）")
        
        # 終了処理
        self._finalize_optimization(self.best_omega, self.best_j, self.best_alpha, 'グリッドサーチ')
        
        return self.best_omega, self.best_j    
    
    def optimize_adaptive_grid(self):
        """適応的グリッドサーチ（粗→細）"""
        n_calls = self._get_opt_config('n_calls')
        n_coarse = max(5, n_calls // 3)
        n_fine = n_calls - n_coarse
        
        if self.optimization_mode not in ['omega_only', 'alpha_only']:
            print("警告: 適応的グリッドは1次元のみ対応。グリッドサーチを実行します。")
            return self.optimize_grid()
        
        # パラメータ設定を取得
        params = self._get_1d_param_settings()
        
        print(f"適応的グリッドサーチ:")
        print(f"  第1段階（粗探索）: {n_coarse}点")
        print(f"  第2段階（細探索）: {n_fine}点")
        if params['fixed_param'] is not None:
            print(f"  {params['fixed_name']}固定値: {params['fixed_param']:.{self.omega_digits}f}")
        
        # 第1段階: 粗いグリッド
        coarse_values = np.linspace(params['param_min'], params['param_max'], n_coarse)
        
        print(f"\n第1段階: 粗探索 [{params['param_min']:.{self.omega_digits}f}, {params['param_max']:.{self.omega_digits}f}]")
        
        try:
            for i, param in enumerate(coarse_values, 1):
                param_rounded = round(param, self.omega_digits)
                print(f"[{i}/{n_coarse}] {params['param_name']} = {param_rounded:.{self.omega_digits}f}")
                self.objective_function([param_rounded])
        except StopIteration:
            print(f"\n最適化が早期終了しました（J値が閾値以下）")
            optimal_omega, optimal_alpha = self._set_optimal_params_by_mode(
                self.best_omega if self.optimization_mode == 'omega_only' else self.best_alpha
            )
            self._finalize_optimization(optimal_omega, self.best_j, optimal_alpha, '適応的グリッドサーチ（早期終了）')
            return optimal_omega, self.best_j
                    
        # 最良点周辺を特定
        best_param = self.best_omega if self.optimization_mode == 'omega_only' else self.best_alpha
        
        print(f"\n第1段階完了: 最良点 {params['param_name']} = {best_param:.{self.omega_digits}f}, J = {self.best_j:.6f}")
        
        # 第2段階: 最良点周辺の細かいグリッド
        search_width = (params['param_max'] - params['param_min']) / (2 * n_coarse)
        fine_min = max(params['param_min'], best_param - search_width)
        fine_max = min(params['param_max'], best_param + search_width)
        
        print(f"\n第2段階: 細探索 [{fine_min:.{self.omega_digits}f}, {fine_max:.{self.omega_digits}f}]")
        
        fine_values = np.linspace(fine_min, fine_max, n_fine)
        
        try:
            for i, param in enumerate(fine_values, 1):
                param_rounded = round(param, self.omega_digits)
                if abs(param_rounded - best_param) < 10 ** (-self.omega_digits):
                    print(f"[{i}/{n_fine}] {params['param_name']} = {param_rounded:.{self.omega_digits}f} (スキップ: 既評価)")
                    continue
                print(f"[{i}/{n_fine}] {params['param_name']} = {param_rounded:.{self.omega_digits}f}")
                self.objective_function([param_rounded])
        except StopIteration:
            print(f"\n最適化が早期終了しました（J値が閾値以下）")
        
        # 終了処理
        optimal_omega, optimal_alpha = self._set_optimal_params_by_mode(
            self.best_omega if self.optimization_mode == 'omega_only' else self.best_alpha
        )
        self._finalize_optimization(optimal_omega, self.best_j, optimal_alpha, '適応的グリッドサーチ')
        
        return optimal_omega, self.best_j
    
    def save_results(self, optimal_omega, optimal_j, optimal_alpha=None):
        """結果をファイルに保存"""
        base_name = Path(self.config['molecule']['gjf_file']).stem
        omega_format = f"{{:.{self.omega_digits}f}}"
        
        with open(f"{base_name}_optimization_results.txt", 'w') as f:
            f.write(f"Executed Script: {os.path.basename(__file__)} (version {__version__})\n\n")
            f.write("Parameter Optimization Results\n")
            f.write("=" * 50 + "\n\n")
            f.write("Features:\n")
            f.write("  - Three optimization modes: omega_only, alpha_only, omega_alpha\n")
            f.write("  - Parameter priority: YAML > GJF IOP > Default\n")
            f.write("  - Corrected J-value calculation using anion HOMO\n\n")
            f.write(f"Molecule: {self.config['molecule']['gjf_file']}\n")
            f.write(f"Optimization mode: {self.optimization_mode}\n")
            
            try:
                temp_calc = DeltaSCFCalculator(
                    self.config['molecule']['gjf_file'], 
                    nproc=1,
                    omega_digits=self.omega_digits
                )
                f.write(f"Functional: {temp_calc.method}\n")
                f.write(f"Basis: {temp_calc.basis}\n")
            except:
                pass
                
            f.write(f"Omega digits: {self.omega_digits}\n")
            f.write(f"Beta mode: {self.beta_mode}\n")
            if self._is_solvent_constraint():
                f.write(f"Epsilon (dielectric constant): {self.epsilon}\n")

            if self.optimization_mode == 'omega_only':
                f.write(f"Optimal omega: {omega_format.format(optimal_omega)}\n")
                if optimal_alpha is not None:
                    f.write(f"Fixed alpha: {omega_format.format(optimal_alpha)}\n")
                else:
                    f.write(f"Fixed alpha: Gaussian default\n")
            elif self.optimization_mode == 'alpha_only':
                if optimal_omega is not None:
                    f.write(f"Fixed omega: {omega_format.format(optimal_omega)}\n")
                else:
                    f.write(f"Fixed omega: Gaussian default\n")
                if optimal_alpha is not None:
                    f.write(f"Optimal alpha: {omega_format.format(optimal_alpha)}\n")
                else:
                    f.write(f"Optimal alpha: optimization failed\n")
            elif self.optimization_mode == 'omega_alpha':
                f.write(f"Optimal omega: {omega_format.format(optimal_omega)}\n")
                if optimal_alpha is not None:
                    f.write(f"Optimal alpha: {omega_format.format(optimal_alpha)}\n")
                else:
                    f.write(f"Optimal alpha: optimization failed\n")
            
            beta_value = self._beta_value(optimal_alpha)
            if beta_value is not None:
                f.write(f"Beta: {beta_value:.{self.omega_digits}f}\n")
            
            f.write(f"Minimum J value: {optimal_j:.6f} eV²\n")
            f.write(f"Total evaluations: {len(self.results_history)}\n\n")
            
            f.write("Evaluation History:\n")
            f.write("-" * 80 + "\n")
            f.write("Omega\t\tAlpha\t\tJ Value\t\tIP (eV)\t\tEA (eV)\t\tHOMO_q0 (eV)\tHOMO_qm1 (eV)\n")
            
            for result in self.results_history:
                omega_str = omega_format.format(result['omega']) if result['omega'] is not None else "default"
                alpha_str = omega_format.format(result['alpha']) if result['alpha'] is not None else "default"
                f.write(f"{omega_str}\t\t"
                       f"{alpha_str}\t\t"
                       f"{result['j_value']:.6f}\t"
                       f"{result['ip_ev']:.3f}\t\t{result['ea_ev']:.3f}\t\t"
                       f"{result['homo_ev_q0']:.3f}\t\t{result['homo_ev_qm1']:.3f}\n")
           
        print(f"\n結果を {base_name}_optimization_results.txt に保存しました")
    
    def plot_results(self, optimal_omega, optimal_j, optimal_alpha=None):
        """結果をPNGファイルに出力"""
        if not self.results_history:
            print("No data available for plotting")
            return
        
        base_name = Path(self.config['molecule']['gjf_file']).stem
        omega_format = f"{{:.{self.omega_digits}f}}"
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
            
            if self.optimization_mode == 'omega_alpha':
                # 2次元最適化の場合
                # Noneを除外
                valid_results = [r for r in self.results_history if r['omega'] is not None and r['alpha'] is not None]
                if not valid_results:
                    print("プロット用の有効なデータがありません")
                    return
                omegas = [r['omega'] for r in valid_results]
                alphas = [r['alpha'] for r in valid_results]
                j_values = [r['j_value'] for r in valid_results]
                
                plt.figure(figsize=(12, 8))
                
                scatter = plt.scatter(omegas, alphas, c=j_values, s=60, alpha=0.8, 
                                    cmap='viridis', edgecolors='black', linewidth=0.5,
                                    norm=LogNorm())
                
                if optimal_alpha is not None:
                    plt.scatter(optimal_omega, optimal_alpha, color='red', s=200, marker='*', 
                               label=f'Optimal (ω={omega_format.format(optimal_omega)}, α={omega_format.format(optimal_alpha)}, J={optimal_j:.6f})', 
                               edgecolors='white', linewidth=2, zorder=5)
                else:
                    plt.scatter(optimal_omega, 0, color='red', s=200, marker='*', 
                               label=f'Optimal (ω={omega_format.format(optimal_omega)}, J={optimal_j:.6f})', 
                               edgecolors='white', linewidth=2, zorder=5)
                
                cbar = plt.colorbar(scatter)
                cbar.set_label('J value (eV²) - Log scale', fontsize=12)
                
                plt.xlabel('ω (Range-separation parameter)', fontsize=12)
                plt.ylabel('α (HF exchange parameter)', fontsize=12)
                plt.title(f'2D Optimization Results (ω+α) - {base_name}', fontsize=14, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                
                output_file = f'{base_name}_omega_alpha_optimization.png'
                
            else:
                # 1次元最適化の場合
                if self.optimization_mode == 'omega_only':
                    # Noneを除外
                    valid_results = [r for r in self.results_history if r['omega'] is not None]
                    if not valid_results:
                        print("プロット用の有効なデータがありません")
                        return
                    param_values = [r['omega'] for r in valid_results]
                    param_name = 'ω (Range-separation parameter)'
                    optimal_param = optimal_omega
                    title_suffix = 'ω optimization'
                elif self.optimization_mode == 'alpha_only':
                    # Noneを除外
                    valid_results = [r for r in self.results_history if r['alpha'] is not None]
                    if not valid_results:
                        print("プロット用の有効なデータがありません")
                        return
                    param_values = [r['alpha'] for r in valid_results]
                    param_name = 'α (HF exchange parameter)'
                    optimal_param = optimal_alpha
                    title_suffix = 'α optimization'
                
                j_values = [r['j_value'] for r in valid_results]
                
                plt.figure(figsize=(10, 6))
                
                plt.scatter(param_values, j_values, alpha=0.7, s=50, color='blue', label='Evaluation points')
                
                if optimal_param is not None:
                    plt.scatter(optimal_param, optimal_j, color='red', s=150, marker='*', 
                               label=f'Optimal point (J={optimal_j:.6f})', 
                               edgecolors='black', linewidth=1)
                
                plt.xlabel(param_name, fontsize=12)
                plt.ylabel('J value (eV²)', fontsize=12)
                plt.title(f'{title_suffix} Results - {base_name}', fontsize=14, fontweight='bold')
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                
                param_range = max(param_values) - min(param_values)
                j_range = max(j_values) - min(j_values)
                plt.xlim(min(param_values) - param_range*0.05, max(param_values) + param_range*0.05)
                plt.ylim(min(j_values) - j_range*0.05, max(j_values) + j_range*0.05)
                
                output_file = f'{base_name}_{self.optimization_mode}_optimization.png'
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Optimization graph saved to {output_file}")
            
        except ImportError:
            print("Warning: matplotlib is not installed. No graph will be generated.")
        except Exception as e:
            print(f"Error during graph generation: {e}")


def create_sample_config():
    """サンプル設定ファイルを作成"""
    with open('omega_optimization_config.yaml', 'w') as f:
        yaml.dump(default_config(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print("サンプル設定ファイル 'omega_optimization_config.yaml' を作成しました")


def submit_optimization_job(config_file, prepare_only=False):
    """最適化全体を1本のジョブとしてRCCSに投入する
    
    g16sub -P（準備のみ）でRCCS用の設定が入ったジョブスクリプトを作らせ、
    その中の g16 を起動する行を、このスクリプトを --in-job で起動する行に書き換えて投入する。
    ジョブ内では3状態のg16を、GAUSS_CDEFのCPUを分けて同時に実行する。
    （方法は shin1koda/rccs-dmf-g16sub を参考にした）
    """
    config_path = Path(config_file).resolve()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    calc = config.get('calculation', {})
    gjf_file = config['molecule']['gjf_file']
    if not Path(gjf_file).exists():
        print(f"エラー: gjfファイルが見つかりません: {gjf_file}（gjfのあるフォルダで実行してください）")
        sys.exit(1)
    
    nprocs = state_nprocs(config)
    ncpus = job_ncpus(config)
    job_walltime = seconds_to_walltime(walltime_to_seconds(calc.get('job_walltime', '72:00:00')))
    
    # g16sub -P でジョブスクリプトを作成（gjfはジョブ名・環境設定のためだけに使う）
    stale = Path(f"{gjf_file}.ap")
    if stale.exists():
        stale.unlink()
    command = ["g16sub", "-P", "-np", str(ncpus), "--walltime", job_walltime]
    command += [str(x) for x in calc.get('g16sub_options', [])]
    command.append(gjf_file)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    if result.returncode != 0:
        print(f"g16sub -P が失敗しました (return code: {result.returncode})")
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    
    job_script = None
    submit_cmd = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Job script"):
            job_script = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("/usr/local/bin/jsub") or stripped.startswith("jsub "):
            submit_cmd = stripped
    if not job_script or not Path(job_script).exists():
        print("エラー: g16sub の出力からジョブスクリプトを見つけられませんでした")
        print(result.stdout)
        sys.exit(1)
    
    # g16 を起動する行を、このスクリプトをジョブ内モードで起動する行に置き換える
    log_file = f"{Path(gjf_file).stem}_opt.log"
    run_line = (f'"{sys.executable}" -u "{Path(__file__).resolve()}" '
                f'--config "{config_path}" --in-job > {log_file} 2>&1')
    lines = Path(job_script).read_text().splitlines()
    replaced = 0
    new_lines = []
    for line in lines:
        if line.strip().startswith("g16 ") or line.strip() == "g16":
            new_lines.append(run_line)
            replaced += 1
        else:
            new_lines.append(line)
    if replaced != 1:
        print(f"エラー: ジョブスクリプト中の g16 の行を特定できませんでした（{replaced}行）: {job_script}")
        sys.exit(1)
    
    # メール通知: true なら開始・終了・中断時に通知（#PBS -m abe）、false なら通知しない
    mail = bool(calc.get('mail', False))
    new_lines = [l for l in new_lines if not l.strip().startswith("#PBS -m")]
    if mail:
        last_pbs = max(i for i, l in enumerate(new_lines) if l.strip().startswith("#PBS"))
        new_lines.insert(last_pbs + 1, "#PBS -m abe")
    Path(job_script).write_text("\n".join(new_lines) + "\n")
    if stale.exists():
        stale.unlink()  # g16subが作った入力ファイルは使わない
    
    print(f"ジョブスクリプトを作成しました: {job_script}")
    breakdown = ' + '.join(f"{s} {nprocs[s]}" for s in STATES)
    if n_parallel_slots(config) > 1:
        print(f"  コア数: {ncpus}（{breakdown} を同時に実行）")
    else:
        print(f"  コア数: {ncpus}（3状態を順番に実行: {breakdown.replace(' + ', ', ')}）")
    print(f"  walltime: {job_walltime}（最適化全体）")
    print(f"  ログ: {log_file}")
    print(f"  メール通知: {'開始・終了時に送信' if mail else 'なし'}")
    
    if prepare_only:
        print("\n--prepare-only のため投入していません。内容を確認して投入する場合:")
        print(f"  {submit_cmd or 'jsub ' + job_script}")
        return
    
    if not submit_cmd:
        print("エラー: g16sub の出力から投入コマンド（jsub）を見つけられませんでした")
        print(result.stdout)
        sys.exit(1)
    submitted = subprocess.run(["bash", "-c", submit_cmd], stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True)
    print(submitted.stdout.strip())
    if submitted.returncode != 0:
        print("エラー: ジョブの投入に失敗しました")
        sys.exit(1)
    print(f"\n投入しました。進み具合は  tail -f {log_file}  で確認できます。")


def main():
    parser = argparse.ArgumentParser(
        description='長距離補正汎関数のパラメーター（ω/α/β）をJ値最小化で最適化（RCCS / g16sub版）')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--config', help='設定ファイルのパス')
    parser.add_argument('--gjf', help='gjfファイルのパス')
    parser.add_argument('--mode', choices=['omega_only', 'alpha_only', 'omega_alpha'], 
                        default='omega_only', help='最適化モード')
    parser.add_argument('--omega-min', type=float, default=0.1, help='ω最小値')
    parser.add_argument('--omega-max', type=float, default=0.5, help='ω最大値')
    parser.add_argument('--alpha-max', type=float, default=0.5, help='α最大値')
    parser.add_argument('--fixed-omega', type=float, help='固定ω値（α最適化時）')
    parser.add_argument('--fixed-alpha', type=float, help='固定α値（ω最適化時）')
    parser.add_argument('--n-calls', type=int, default=20, help='最大評価回数')
    parser.add_argument('--nproc', type=int, default=4, help='プロセッサ数')
    parser.add_argument('--parallel', type=int, default=1, help='最大並列ジョブ数')
    parser.add_argument('--create-config', action='store_true', help='サンプル設定ファイルを作成')
    parser.add_argument('--submit', action='store_true',
                        help='最適化全体を1本のジョブとして投入（フロントエンドで待機しない）')
    parser.add_argument('--prepare-only', action='store_true',
                        help='--submit と併用: ジョブスクリプトを作るだけで投入しない')
    parser.add_argument('--in-job', action='store_true', help=argparse.SUPPRESS)  # --submitのジョブ内で使用
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    if args.submit or args.prepare_only:
        if not args.config:
            print("エラー: --submit には --config が必要です")
            sys.exit(1)
        submit_optimization_job(args.config, prepare_only=args.prepare_only)
        return
    
    # 設定の準備
    if args.config:
        overrides = {'calculation.launcher': 'direct'} if args.in_job else {}
        optimizer = OmegaOptimizer(args.config, **overrides)
    else:
        if not args.gjf:
            print("エラー: --gjf または --config が必要です")
            sys.exit(1)
        
        # コマンドライン引数から設定を作成
        config_kwargs = {
            'molecule.gjf_file': args.gjf,
            'optimization.optimization_mode': args.mode,
            'optimization.omega_min': args.omega_min,
            'optimization.omega_max': args.omega_max,
            'optimization.alpha_max': args.alpha_max,
            'optimization.n_calls': args.n_calls,
            'calculation.nproc_per_job': args.nproc,
            'optimization.max_parallel_jobs': args.parallel
        }
        
        if args.fixed_omega is not None:
            config_kwargs['optimization.fixed_omega'] = args.fixed_omega
        if args.fixed_alpha is not None:
            config_kwargs['optimization.fixed_alpha'] = args.fixed_alpha
            
        optimizer = OmegaOptimizer(**config_kwargs)
    
    # 最適化の実行
    try:
        optimal_omega, optimal_j = optimizer.optimize()
    except Exception as e:
        print(f"最適化中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    
    