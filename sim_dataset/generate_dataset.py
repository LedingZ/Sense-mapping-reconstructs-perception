import os
import json
import numpy as np
import soundfile as sf
import pyroomacoustics as pra

# ----------------------------
# Config
# ----------------------------
OUT_DIR = "sim_dataset"
FS = 16000
DURATION_S = 6.0          # 每条样本时长
N_SAMPLES = 50            # 生成多少条混合样本
ROOM_DIMS = (8.0, 6.0, 3.0)  # 房间尺寸 (x,y,z) meters
RT60 = 0.35               # 混响强度（越大越混响）；想更“干”就设 0.1 或 0.0
SEED = 42

rng = np.random.default_rng(SEED)


# ----------------------------
# Helpers
# ----------------------------
def normalize(x, peak=0.98):
    m = np.max(np.abs(x)) + 1e-12
    return (x / m) * peak

def random_point_in_room(margin=0.5):
    # 随机点，离墙至少 margin 米
    x = rng.uniform(margin, ROOM_DIMS[0] - margin)
    y = rng.uniform(margin, ROOM_DIMS[1] - margin)
    z = rng.uniform(1.0, 1.8)  # 大概“人耳高度”
    return np.array([x, y, z], dtype=float)

def make_five_sources(fs, n):
    """生成 5 种“明显不同”的声音（无需外部音频文件）"""
    t = np.arange(n) / fs

    # 1) 低频正弦 + 轻微颤音
    s1 = 0.6*np.sin(2*np.pi*(130 + 10*np.sin(2*np.pi*0.7*t))*t)

    # 2) 高频正弦（更尖）
    s2 = 0.4*np.sin(2*np.pi*1200*t)

    # 3) 线性扫频 chirp（中频穿行）
    f0, f1 = 200, 2500
    k = (f1 - f0) / (t[-1] + 1e-9)
    phase = 2*np.pi*(f0*t + 0.5*k*t**2)
    s3 = 0.45*np.sin(phase)

    # 4) 噪声（更像环境底噪）
    s4 = 0.15*rng.standard_normal(n)

    # 5) click train（像脚步/敲击）
    s5 = np.zeros(n, dtype=float)
    click_idx = (np.arange(0.3, DURATION_S, 0.6) * fs).astype(int)
    for i in click_idx:
        if 0 <= i < n:
            s5[i:i+80] += np.hanning(min(80, n-i)) * 0.9

    sources = [s1, s2, s3, s4, s5]
    sources = [normalize(s, peak=0.7) for s in sources]
    return sources

def build_room():
    # 用 RT60 估算吸音参数（更真实）；想无混响可以直接 pra.ShoeBox(..., max_order=0)
    e_absorption, max_order = pra.inverse_sabine(RT60, ROOM_DIMS[:2])  # 用 xy 尺寸估
    mat = pra.Material(e_absorption)

    room = pra.ShoeBox(
        ROOM_DIMS,
        fs=FS,
        materials=mat,
        max_order=max_order
    )
    return room


# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(os.path.join(OUT_DIR, "sources"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "mixes"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "meta"), exist_ok=True)

    n = int(FS * DURATION_S)

    # 固定 5 个源音色（每条样本共用），也可以改成每条样本重新生成
    base_sources = make_five_sources(FS, n)

    # 保存“原始干声源”（可选）
    for k, s in enumerate(base_sources, start=1):
        sf.write(os.path.join(OUT_DIR, "sources", f"src{k}_dry.wav"), s, FS)

    meta_all = []

    for i in range(N_SAMPLES):
        room = build_room()

        # 每条样本：随机 5 个声源位置 + 1 个麦克风位置
        src_positions = [random_point_in_room() for _ in range(5)]
        mic_position = random_point_in_room()

        # 加麦克风（单通道）
        room.add_microphone_array(pra.MicrophoneArray(mic_position.reshape(3, 1), FS))

        # 加声源
        for k in range(5):
            # 每条样本给每个源随机一个增益，模拟音量差异
            gain = rng.uniform(0.4, 1.0)
            sig = base_sources[k] * gain
            room.add_source(src_positions[k], signal=sig)

        # 跑仿真（生成混合录音）
        room.simulate()
        mix = room.mic_array.signals[0]  # shape (n,)

        mix = normalize(mix, peak=0.98)

        # 输出文件
        mix_name = f"mix_{i:04d}.wav"
        mix_path = os.path.join(OUT_DIR, "mixes", mix_name)
        sf.write(mix_path, mix, FS)

        # 存 metadata（位置、房间、RT60、文件名）
        meta = {
            "id": i,
            "fs": FS,
            "duration_s": DURATION_S,
            "room_dims_xyz": ROOM_DIMS,
            "rt60": RT60,
            "mic_xyz": mic_position.tolist(),
            "src_xyz_list": [p.tolist() for p in src_positions],
            "mix_wav": mix_path,
        }

        meta_all.append(meta)

        # 每条样本单独一份 json
        with open(os.path.join(OUT_DIR, "meta", f"mix_{i:04d}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if (i + 1) % 10 == 0:
            print(f"[OK] generated {i+1}/{N_SAMPLES}")

    # 总表
    with open(os.path.join(OUT_DIR, "meta", "all_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)

    print("\nDONE.")
    print(f"- mixes: {OUT_DIR}/mixes/")
    print(f"- meta : {OUT_DIR}/meta/")
    print(f"- dry sources (optional): {OUT_DIR}/sources/")

if __name__ == "__main__":
    main()
