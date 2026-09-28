"""
Aproveita as caixas de um dataset que NÃO tem os dígitos identificados
(ex.: exportação Roboflow com uma única classe "hidrometro").

Para cada foto do dataset:
  * acha a foto ORIGINAL correspondente em imagens_treino/ (mesmo que o
    Roboflow tenha redimensionado) -> usa a original, em resolução cheia;
    se não achar, copia a foto do dataset para imagens_treino/
  * 1 caixa  -> vira os 4 cantos do visor
  * 3+ caixas -> são os dígitos individuais (ordenados da esquerda p/ direita)
  * grava tudo pré-preenchido para o rotular.py

Depois é só rodar  python rotular.py : as caixas já aparecem, basta digitar a
leitura e teclar ENTER.

Uso:  python importar_roboflow.py [pasta]      (padrão: dataset/)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from leitor import ler_imagem, listar_imagens, mapear_classes
from preparar_dataset import achar_label, achar_nomes_classes

RAIZ = Path(__file__).resolve().parent
TREINO = RAIZ / "imagens_treino"
ARQ_ANOT = RAIZ / "dataset" / "proprio" / "anotacoes.json"


def dhash(img: np.ndarray) -> np.ndarray:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (17, 16), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def ler_caixas(lbl: Path) -> list[tuple[float, float, float, float]]:
    """Caixas normalizadas (x1, y1, x2, y2); polígonos viram caixa."""
    caixas = []
    for linha in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
        v = linha.split()
        if len(v) < 5:
            continue
        n = [float(x) for x in v[1:]]
        if len(n) == 4:
            xc, yc, w, h = n
            caixas.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
        elif len(n) >= 6 and len(n) % 2 == 0:
            xs, ys = n[0::2], n[1::2]
            caixas.append((min(xs), min(ys), max(xs), max(ys)))
    return caixas


def main():
    origem = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "dataset"
    TREINO.mkdir(exist_ok=True)
    ARQ_ANOT.parent.mkdir(parents=True, exist_ok=True)
    anot = json.loads(ARQ_ANOT.read_text(encoding="utf-8")) if ARQ_ANOT.exists() else {}

    originais = {}
    for p in listar_imagens(TREINO):
        img = ler_imagem(p)
        if img is not None:
            originais[p.name] = (dhash(img), img.shape[:2])

    n_visor = n_digitos = n_novas = n_ja = 0
    for img_path in listar_imagens(origem):
        if "proprio" in img_path.parts:
            continue
        lbl = achar_label(img_path)
        if lbl is None:
            continue
        nomes = achar_nomes_classes(lbl.parent) or {}
        if mapear_classes(nomes)[0]:
            continue          # já tem dígitos identificados: preparar_dataset usa direto
        caixas = ler_caixas(lbl)
        if not caixas:
            continue
        img = ler_imagem(img_path)
        if img is None:
            continue

        # procura a original (mesma foto, mesmo que esticada/redimensionada)
        h = dhash(img)
        nome, dist = None, 999
        for n, (ho, _) in originais.items():
            d = int((ho != h).sum())
            if d < dist:
                nome, dist = n, d
        if dist > 12:
            nome = f"rf_{img_path.stem.split('.rf.')[0][:50]}{img_path.suffix.lower()}"
            shutil.copy2(img_path, TREINO / nome)
            originais[nome] = (h, img.shape[:2])
            n_novas += 1
        H, W = originais[nome][1]

        if anot.get(nome, {}).get("leitura") or anot.get(nome, {}).get("descartada"):
            n_ja += 1
            continue
        px = [(x1 * W, y1 * H, x2 * W, y2 * H) for x1, y1, x2, y2 in caixas]
        if len(px) >= 3:        # dígitos individuais
            px.sort(key=lambda b: (b[0] + b[2]) / 2)
            x1 = min(b[0] for b in px); y1 = min(b[1] for b in px)
            x2 = max(b[2] for b in px); y2 = max(b[3] for b in px)
            anot[nome] = {"caixas": [list(b) for b in px],
                          "pontos": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "importada": True}
            n_digitos += 1
        else:                   # visor (se vierem 2 caixas, usa a união)
            x1 = min(b[0] for b in px); y1 = min(b[1] for b in px)
            x2 = max(b[2] for b in px); y2 = max(b[3] for b in px)
            anot[nome] = {"pontos": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "importada": True}
            n_visor += 1

    ARQ_ANOT.write_text(json.dumps(anot, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(ARQ_ANOT.parent / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"names": [str(i) for i in range(10)] + ["visor"]}, f)
    print(f"Visor pré-marcado:            {n_visor} fotos")
    print(f"Dígitos individuais marcados: {n_digitos} fotos")
    print(f"Fotos novas copiadas para imagens_treino/: {n_novas}")
    if n_ja:
        print(f"Já tinham leitura (mantidas): {n_ja}")
    print("\nAgora rode:  python rotular.py   -> digite a leitura de cada foto + ENTER")


if __name__ == "__main__":
    main()
