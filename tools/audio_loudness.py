"""Measure the loudness of every game sound and write the level manifest.

Loudness is ITU-R BS.1770 (ffmpeg's ebur128 filter) on the mono downmix,
which is what the battle mixer plays. A one-shot is measured by its
loudest 400 ms window (maximum momentary loudness); a sustained clip
(a loop, a bed, a group sheet, anything 3 s or longer) by its integrated
loudness. Each entry also carries the true peak, the length and the gain
that brings the clip to its target.

Run from the repo root:

    python3 tools/audio_loudness.py            # writes assets/audio_levels.json
    python3 tools/audio_loudness.py --table    # also prints every clip

Needs ffmpeg and ffprobe on the PATH.
"""
import json
import os
import re
import subprocess
import sys

ASSETS = "assets"
OUT = os.path.join(ASSETS, "audio_levels.json")
EXTENSIONS = (".mp3", ".wav", ".ogg")
# Targets in LUFS: one-shots by their loudest 400 ms, sustained clips by
# their integrated loudness.
TARGET_ONE_SHOT = -20.0
TARGET_SUSTAINED = -23.0
SUSTAINED_SECONDS = 3.0
SUSTAINED_NAMES = ("loop", "bed_", "wash", "drums")

MOMENTARY = re.compile(r" M:\s*(-?[\d.]+)")
INTEGRATED = re.compile(r"^\s*I:\s*(-?[\d.]+) LUFS", re.M)
TRUE_PEAK = re.compile(r"^\s*Peak:\s*(-?[\d.inf]+) dBFS", re.M)


def probe(path):
    """Return (duration in seconds, channel count)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels:format=duration",
         "-of", "default=noprint_wrappers=1", path],
        capture_output=True, text=True).stdout
    fields = dict(line.split("=", 1) for line in out.split() if "=" in line)
    return float(fields.get("duration", 0.0)), int(fields.get("channels", 1))


def measure(path, channels):
    """Return (max momentary, integrated, true peak) of the mono downmix."""
    mono = "pan=mono|c0=0.5*c0+0.5*c1," if channels >= 2 else ""
    # The padding lets the last 400 ms window of a short clip complete.
    chain = f"{mono}apad=pad_dur=0.4,ebur128=peak=true:framelog=verbose"
    err = subprocess.run(
        ["ffmpeg", "-nostats", "-hide_banner", "-v", "verbose", "-i", path,
         "-af", chain, "-f", "null", "-"],
        capture_output=True, text=True).stderr
    momentary = [float(m) for m in MOMENTARY.findall(err) if float(m) > -120.0]
    integrated = INTEGRATED.findall(err)
    peak = TRUE_PEAK.findall(err)
    return (max(momentary) if momentary else None,
            float(integrated[-1]) if integrated else None,
            float(peak[-1]) if peak and peak[-1] != "-inf" else None)


def main():
    clips = {}
    # Tracked files only: local extras never reach the manifest.
    tracked = subprocess.run(["git", "ls-files", ASSETS],
                             capture_output=True, text=True).stdout.split("\n")
    for path in sorted(p for p in tracked if p.lower().endswith(EXTENSIONS)):
        name = os.path.basename(path)
        rel = os.path.relpath(path, ASSETS)
        duration, channels = probe(path)
        momentary, integrated, peak = measure(path, channels)
        sustained = duration >= SUSTAINED_SECONDS or any(
            key in name.lower() for key in SUSTAINED_NAMES)
        loudness = integrated if sustained else momentary
        target = TARGET_SUSTAINED if sustained else TARGET_ONE_SHOT
        clips[rel] = {
            "measure": "integrated" if sustained else "momentary_max",
            "lufs": None if loudness is None else round(loudness, 1),
            "true_peak_db": None if peak is None else round(peak, 1),
            "seconds": round(duration, 2),
            "gain_db": None if loudness is None else round(target - loudness, 1),
        }
    manifest = {
        "about": "BS.1770 loudness of the mono downmix per clip; written by tools/audio_loudness.py",
        "target_one_shot_lufs": TARGET_ONE_SHOT,
        "target_sustained_lufs": TARGET_SUSTAINED,
        "clips": dict(sorted(clips.items())),
    }
    with open(OUT, "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"wrote {OUT}: {len(clips)} clips")
    if "--table" in sys.argv:
        for rel, c in sorted(clips.items()):
            print(f"{c['lufs']!s:>6} LUFS  {c['gain_db']!s:>6} dB  "
                  f"peak {c['true_peak_db']!s:>5}  {c['seconds']:5.2f} s  "
                  f"{c['measure']:13s}  {rel}")


if __name__ == "__main__":
    main()
