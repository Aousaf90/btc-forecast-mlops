#!/usr/bin/env python3
"""Render assignment CLI transcripts as terminal-style PNG screenshots."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots"
LOGS = OUT / "logs"

SHOTS = [
    ("D_kubectl_overview.png", "kubectl get pods,svc,hpa,ingress,pvc", LOGS / "D_summary.txt"),
    ("D5_hpa_scale.png", "kubectl describe hpa  (2 → 6 → 8 → scale-down)", LOGS / "D5_hpa_scaled.txt"),
    ("D5_hpa_events.png", "HPA events after load + scale-down", LOGS / "D5_hpa_now.txt"),
    ("E1_self_healing.png", "kubectl delete pod  — ReplicaSet replacement", LOGS / "E1_self_healing.txt"),
    ("E2_live_predict.png", "Live Kubernetes API  /health /metrics /predict", LOGS / "E2_live_k8s_api.txt"),
    ("E2_train_job.png", "Job btc-forecast-train  (fetch → train → reload)", LOGS / "E2_train_job.txt"),
    ("C1_airflow_dag.png", "Airflow DAG btc_etl_pipeline  — all 4 tasks success", LOGS / "C1_airflow_dag.txt"),
    ("C2_kfp_compile.png", "Kubeflow pipeline compile", LOGS / "C2_kfp_compile.txt"),
    ("A3_pytest.png", "pytest  — 85 passed", LOGS / "A3_pytest.txt"),
    ("A4_training.png", "Hold-out evaluation + 7-day forecast", LOGS / "A4_training.txt"),
    ("B3_github_actions.png", "GitHub Actions CI/CD runs", LOGS / "B3_github_actions.txt"),
]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render(title: str, text: str, max_lines: int = 42) -> Image.Image:
    lines = text.replace("\t", "    ").splitlines()
    if len(lines) > max_lines:
        head, tail = max_lines // 2 - 1, max_lines // 2 - 1
        lines = lines[:head] + ["    …"] + lines[-tail:]
    mono = font(13)
    title_f = font(14)
    pad_x, pad_y = 18, 16
    header_h = 36
    line_h = 18
    widths = [mono.getlength(ln) for ln in lines] or [400]
    w = int(min(1100, max(720, max(widths) + pad_x * 2 + 8)))
    h = header_h + pad_y + line_h * max(len(lines), 8) + pad_y
    img = Image.new("RGB", (w, h), "#0d1117")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, header_h], fill="#161b22")
    draw.ellipse([12, 12, 24, 24], fill="#ff5f56")
    draw.ellipse([30, 12, 42, 24], fill="#ffbd2e")
    draw.ellipse([48, 12, 60, 24], fill="#27c93f")
    draw.text((76, 10), title, font=title_f, fill="#c9d1d9")
    y = header_h + 8
    for ln in lines:
        draw.text((pad_x, y), ln[:160], font=mono, fill="#e6edf3")
        y += line_h
    return img


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, title, src in SHOTS:
        if not src.exists():
            print(f"skip missing {src}")
            continue
        img = render(title, src.read_text(encoding="utf-8", errors="replace"))
        dest = OUT / name
        img.save(dest, "PNG", optimize=True)
        print(f"wrote {dest.relative_to(ROOT)}  {img.size[0]}x{img.size[1]}")


if __name__ == "__main__":
    main()
