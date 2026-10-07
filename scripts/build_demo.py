"""Monta a demonstracao dos dois videos anotados lado a lado, sem recortar a cena."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(ROOT / "docs/results/video_1_archery_anotado.mp4"),
        "-i", str(ROOT / "docs/results/video_2_theatre_anotado.mp4"),
        "-filter_complex",
        ("[0:v]pad=320:240:0:0:black,drawtext=text='V1 - ARCO E FLECHA':x=8:y=h-18:fontsize=11:fontcolor=white:box=1:boxcolor=black@0.5[a];"
        "[1:v]pad=320:240:0:30:black,drawtext=text='V2 - PALCO / FLAMENCO':x=8:y=h-18:fontsize=11:fontcolor=white:box=1:boxcolor=black@0.5[b];"
        "[a][b]hstack=inputs=2[v]"),
        "-map", "[v]", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(ROOT / "docs/demo_cp5.mp4"),
    ], check=True)


if __name__ == "__main__":
    main()
