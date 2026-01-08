import json
import numpy as np
import soundfile as sf
import pyroomacoustics as pra
from pathlib import Path

FS = 16000
DURATION_S = 3.0
N = int(FS * DURATION_S)

OUT = Path("sim_dataset")
(OUT/"mixes").mkdir(parents=True, exist_ok=True)
(OUT/"meta").mkdir(parents=True, exist_ok=True)

ROOM = [60, 60, 3]
Z = 1.5
SPACING = 10

sources = [
    [10, 10, Z],
    [20, 10, Z],
    [30, 10, Z],
    [10, 20, Z],
    [20, 20, Z],
]

# 5-mic cross layout
cx, cy = 30, 30
mic_positions = [
    [cx, cy, Z],
    [cx+SPACING, cy, Z],
    [cx-SPACING, cy, Z],
    [cx, cy+SPACING, Z],
    [cx, cy-SPACING, Z],
]
mic = np.array(mic_positions).T   # (3,5)

def make_sources():
    t = np.arange(N) / FS
    s1 = 0.6*np.sin(2*np.pi*130*t)
    s2 = 0.4*np.sin(2*np.pi*1200*t)
    s3 = 0.45*np.sin(2*np.pi*(200*t + 0.5*(2000/(t[-1]+1e-9))*t**2))
    s4 = 0.15*np.random.randn(N)
    s5 = np.zeros(N); s5[0:80] = np.hanning(80)*0.9
    return [s1, s2, s3, s4, s5]

room = pra.ShoeBox(ROOM, fs=FS, max_order=0)
room.add_microphone_array(pra.MicrophoneArray(mic, FS))

sig = make_sources()
for i, pos in enumerate(sources):
    room.add_source(pos, signal=sig[i])

room.simulate()

signals = room.mic_array.signals  # (5, N)
print("signals shape:", signals.shape)

# save per-mic wavs
mic_wavs = []
for i in range(signals.shape[0]):
    x = signals[i]
    x = x/(np.max(np.abs(x))+1e-12)*0.98
    name = f"mix_fixed_mic{i}.wav"
    sf.write(OUT/"mixes"/name, x, FS)
    mic_wavs.append(str(OUT/"mixes"/name))

# save a mixed wav (average)
mix = np.mean(signals, axis=0)
mix = mix/(np.max(np.abs(mix))+1e-12)*0.98
mix_name = "mix_fixed.wav"
sf.write(OUT/"mixes"/mix_name, mix, FS)

meta = {
    "room_dims_xyz": ROOM,
    "mic_xyz_list": mic_positions,
    "src_xyz_list": sources,
    "mic_wavs": mic_wavs,
    "mix_wav": str(OUT/"mixes"/mix_name),
    "spacing_m": SPACING
}
(OUT/"meta"/"mix_fixed.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("DONE.")
print("mics:", meta["mic_xyz_list"])
for k, p in enumerate(meta["src_xyz_list"]):
    print(f"src{k}:", p)

