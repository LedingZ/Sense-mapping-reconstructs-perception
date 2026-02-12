#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soundset/run_sim.py  （分发阶段：sinput -> speakers/* -> status/ready.txt）

需求实现（按你最后确认的版本）：
1) 脚本放在：sound_proj/soundset/run_sim.py
2) 使用已有虚拟环境 sound_proj/venv（运行时自行激活）
3) 扫描 sound_proj/sinput 下所有音频文件（仅 .mp3 / .wav）
4) 对每个文件做特征分析（librosa），映射到 (x,y) ∈ {-2..2}^2
5) 将文件移动到 sound_proj/speakers/x±n_y±m/（同名冲突自动改名 __1, __2...）
6) 当 sinput 中所有 .mp3/.wav 都被处理并移走后，在 sound_proj/status 生成空白文件：
      ready.txt
   供下一段 watcher 作为触发信号。
7) 同时写一份日志：sound_proj/status/dispatch_log.json（便于排查映射与错误）

运行（在 sound_proj/soundset 下）：
  python run_sim.py
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa


GridCoord = Tuple[int, int]  # (x,y) where x,y in {-2..2}


@dataclass
class Config:
    target_sr: int = 48000

    # 归一化范围（经验值：适合常见音乐/环境音；后续可根据你的素材调整）
    centroid_hz_lo: float = 200.0
    centroid_hz_hi: float = 5000.0

    onset_lo: float = 0.0
    onset_hi: float = 10.0

    flatness_lo: float = 0.05
    flatness_hi: float = 0.8


# ✅ 只支持你要求的两种“常见音乐格式”
AUDIO_EXTS = {".wav", ".mp3"}


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def normalize01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def parse_grid_folder_name(name: str) -> GridCoord | None:
    """
    解析文件夹名：x-2_y+1、x+0_y-1
    """
    try:
        parts = name.split("_")
        if len(parts) != 2:
            return None
        x_str = parts[0].lstrip("x")
        y_str = parts[1].lstrip("y")
        x = int(x_str)
        y = int(y_str)
        if x < -2 or x > 2 or y < -2 or y > 2:
            return None
        return (x, y)
    except Exception:
        return None


def grid_folder_name(x: int, y: int) -> str:
    """
    统一生成文件夹名：x-2_y+1 / x+0_y-1
    """
    return f"x{x:+d}_y{y:+d}"


def ensure_speaker_folders_exist(speakers_dir: Path) -> Dict[GridCoord, Path]:
    """
    确认 speakers/ 下 25 个坐标文件夹都存在，并返回 coord->Path 映射。
    如果你已经全建好了，这里不会影响现有文件夹；缺的才会补齐。
    """
    coord_map: Dict[GridCoord, Path] = {}

    if not speakers_dir.exists():
        raise FileNotFoundError(f"Missing speakers dir: {speakers_dir}")

    # 读已有的
    for p in speakers_dir.iterdir():
        if p.is_dir():
            coord = parse_grid_folder_name(p.name)
            if coord is not None:
                coord_map[coord] = p

    # 补齐缺的
    for y in range(-2, 3):
        for x in range(-2, 3):
            coord = (x, y)
            if coord not in coord_map:
                target = speakers_dir / grid_folder_name(x, y)
                target.mkdir(parents=True, exist_ok=True)
                coord_map[coord] = target

    return coord_map


def extract_features(y: np.ndarray, sr: int, cfg: Config) -> Dict[str, float]:
    """
    最小但稳定的一组特征（“可直接分析”的一阶信号描述符）：
      - spectral centroid（亮度）
      - onset strength mean（瞬态/活动）
      - spectral flatness（噪声/纹理）
      - harmonic ratio（HPSS harmonic能量占比，粗略“可辨识/音高”）

    输出同时带归一化值，便于日志/调参。
    """
    # y: mono
    y = librosa.util.normalize(y)

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    onset = float(np.mean(librosa.onset.onset_strength(y=y, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    y_h, y_p = librosa.effects.hpss(y)
    harm_e = float(np.mean(y_h**2))
    perc_e = float(np.mean(y_p**2))
    harm_ratio = harm_e / (harm_e + perc_e + 1e-9)

    rms = float(np.mean(librosa.feature.rms(y=y)))

    # 归一化
    centroid_n = normalize01(centroid, cfg.centroid_hz_lo, cfg.centroid_hz_hi)
    onset_n = normalize01(onset, cfg.onset_lo, cfg.onset_hi)
    flatness_n = normalize01(flatness, cfg.flatness_lo, cfg.flatness_hi)
    harm_n = float(np.clip(harm_ratio, 0.0, 1.0))

    clarity = 0.65 * harm_n + 0.35 * (1.0 - flatness_n)

    return {
        "centroid": centroid,
        "onset": onset,
        "flatness": flatness,
        "harm_ratio": harm_ratio,
        "rms": rms,
        "centroid_n": centroid_n,
        "onset_n": onset_n,
        "flatness_n": flatness_n,
        "harm_n": harm_n,
        "clarity": clarity,
    }


def map_to_grid(feat: Dict[str, float]) -> GridCoord:
    """
    连续特征 -> 离散网格（中心 (0,0)）

      X: centroid_n  暗->左(-2), 亮->右(+2)
      Y: onset_n     平->下(-2), 瞬态->上(+2)

    输出：x,y ∈ {-2,-1,0,1,2}
    """
    x = int(round(-2 + 4 * feat["centroid_n"]))
    y = int(round(-2 + 4 * feat["onset_n"]))
    return (clamp_int(x, -2, 2), clamp_int(y, -2, 2))


def atomic_touch_empty(path: Path) -> None:
    """
    原子方式创建/覆盖一个空文件（ready.txt）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(b"")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def safe_move(src: Path, dst: Path) -> Path:
    """
    安全移动：若目标已存在则自动改名，避免覆盖。
    返回最终落地路径。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return dst

    stem = dst.stem
    suffix = dst.suffix
    i = 1
    while True:
        alt = dst.with_name(f"{stem}__{i}{suffix}")
        if not alt.exists():
            shutil.move(str(src), str(alt))
            return alt
        i += 1


def list_audio_files(dir_path: Path) -> List[Path]:
    return sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS],
        key=lambda p: p.stat().st_mtime,
    )


def main() -> None:
    cfg = Config()

    # ✅ 脚本在 soundset/ 内，但项目根目录是上一层 sound_proj/
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # sound_proj/

    SINPUT = PROJECT_ROOT / "sinput"
    SPEAKERS = PROJECT_ROOT / "speakers"
    STATUS = PROJECT_ROOT / "status"

    READY_FILE = STATUS / "ready.txt"  # ✅ 你要求必须带 .txt

    if not SINPUT.exists():
        raise FileNotFoundError(f"Missing sinput dir: {SINPUT}")
    if not SPEAKERS.exists():
        raise FileNotFoundError(f"Missing speakers dir: {SPEAKERS}")

    coord_to_dir = ensure_speaker_folders_exist(SPEAKERS)

    files = list_audio_files(SINPUT)

    log_rows = []
    for f in files:
        try:
            # 用 librosa 读取（mp3/wav 都可），统一到 target_sr，输出 mono
            y, sr = librosa.load(f, sr=cfg.target_sr, mono=True)

            feat = extract_features(y, sr, cfg)
            gx, gy = map_to_grid(feat)

            target_dir = coord_to_dir[(gx, gy)]
            final_path = safe_move(f, target_dir / f.name)

            log_rows.append(
                {
                    "file": f.name,
                    "moved_to": str(final_path.relative_to(PROJECT_ROOT)),
                    "mapped_x": gx,
                    "mapped_y": gy,
                    "centroid": feat["centroid"],
                    "onset": feat["onset"],
                    "flatness": feat["flatness"],
                    "harm_ratio": feat["harm_ratio"],
                    "clarity": feat["clarity"],
                }
            )
        except Exception as e:
            log_rows.append({"file": f.name, "error": repr(e)})

    # 写日志（建议保留：用于验证映射/排错）
    STATUS.mkdir(parents=True, exist_ok=True)
    with open(STATUS / "dispatch_log.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "processed_count": len(files),
                "ok_count": sum(1 for r in log_rows if "error" not in r),
                "error_count": sum(1 for r in log_rows if "error" in r),
                "rows": log_rows,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    # ✅ sinput 清空（至少没有 .mp3/.wav 残留）才写 ready.txt
    remaining = list_audio_files(SINPUT)
    if len(remaining) == 0:
        atomic_touch_empty(READY_FILE)


if __name__ == "__main__":
    main()

