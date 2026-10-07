"""Valida a integridade dos arquivos de entrega, sem baixar ou executar redes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
from pypdf import PdfReader

from action_vision.processing import temporal_windows

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert len(PdfReader(ROOT / 'docs/relatorio_cp5.pdf').pages) == 3, 'O PDF deve ter tres paginas.'
    manifest = json.loads((ROOT / 'docs/results/reproducao.json').read_text())
    for stem in ('video_1_archery', 'video_2_theatre'):
        source = ROOT / f'assets/videos/{stem}.mp4'
        expected = manifest['input_sha256'][f'assets/videos/{stem}.mp4']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, f'Entrada alterada: {stem}'
        with (ROOT / f'docs/results/{stem}_frames.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        predictions = json.loads((ROOT / f'docs/results/{stem}_janelas.json').read_text())
        assert len(rows) == 80
        assert [(p['frame_inicial'], p['frame_final']+1) for p in predictions] == temporal_windows(80, 32, 16)
        for index, row in enumerate(rows):
            assert row['acao']
            assert int(row['frame_inicial_janela']) <= index <= int(row['frame_final_janela'])
            matching = [p for p in predictions if p['frame_inicial'] == int(row['frame_inicial_janela'])]
            assert matching[0]['acao'] == row['acao']
            assert abs(matching[0]['confianca'] - float(row['confianca_slowfast'])) <= 0.0000051
        for stage in ('anotado', 'yolo', 'unet', 'slowfast'):
            parent = ROOT / 'docs/results' if stage == 'anotado' else ROOT / 'docs/results/isolados'
            video = cv2.VideoCapture(str(parent / f'{stem}_{stage}.mp4'))
            assert video.isOpened() and int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == len(rows)
            video.release()
        for i in (1, 2, 3):
            assert (ROOT / f'docs/images/evidencias/{stem}_anotado_momento_{i}.jpg').is_file()
        comparisons = json.loads((ROOT / f'docs/results/isolados/{stem}_comparacao.json').read_text())
        assert len(comparisons) == len(predictions)
        assert all(item['origem_caixas_recorte'] == 'pipeline integrado' for item in comparisons)
    demo = cv2.VideoCapture(str(ROOT / 'docs/demo_cp5.mp4'))
    assert demo.isOpened() and int(demo.get(cv2.CAP_PROP_FRAME_COUNT)) == 80
    demo.release()
    print('CP5: PDF com 3 paginas, 2 videos, 6 evidencias, 8 janelas, dados e videos consistentes.')


if __name__ == '__main__':
    main()
