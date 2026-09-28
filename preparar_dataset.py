"""
Prepara o(s) dataset(s) para treino.

Coloque seus datasets (um ou vários) dentro da pasta  dataset/  em
qualquer um destes formatos:

  - Exportação Roboflow "YOLOv8/YOLO11" (train/ valid/ test/ + data.yaml)
  - images/ + labels/ (com ou sem subpastas train/val)
  - imagens e .txt na mesma pasta (+ classes.txt ou data.yaml)

O script:
  * encontra todas as imagens e seus rótulos (.txt formato YOLO)
  * unifica as classes: 0..9 = dígitos, 10 = visor/contador (se existir)
  * converte polígonos (segmentação / OBB) em caixas
  * mantém a divisão train/val/test original ou divide 80/15/5
    (fotos do mesmo hidrômetro ficam sempre no mesmo grupo)
  * cria imagens sintéticas trocando os dígitos de lugar dentro do visor
    (só no treino), equilibrando os dígitos raros
  * gera  dataset_preparado/data.yaml  pronto para o treinar.py

Uso:  python preparar_dataset.py [--sinteticas 10]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import cv2
import yaml

from leitor import EXTENSOES_IMAGEM, ler_imagem, mapear_classes, salvar_imagem

RAIZ = Path(__file__).resolve().parent
ORIGEM = RAIZ / "dataset"
DESTINO = RAIZ / "dataset_preparado"
SEMENTE = 42

CLASSES_FINAIS = [str(i) for i in range(10)] + ["visor"]
ID_VISOR = 10


def achar_nomes_classes(pasta: Path) -> dict[int, str] | None:
    """Procura data.yaml / classes.txt subindo a partir da pasta."""
    p = pasta
    while True:
        for nome in ("data.yaml", "data.yml", "dataset.yaml"):
            f = p / nome
            if f.exists():
                try:
                    d = yaml.safe_load(f.read_text(encoding="utf-8"))
                    n = d.get("names")
                    if isinstance(n, list):
                        return dict(enumerate(n))
                    if isinstance(n, dict):
                        return {int(k): v for k, v in n.items()}
                except Exception as e:
                    print(f"  aviso: não consegui ler {f}: {e}")
        for nome in ("classes.txt", "obj.names"):
            f = p / nome
            if f.exists():
                linhas = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
                return dict(enumerate(linhas))
        if p == ORIGEM or p.parent == p:
            return None
        p = p.parent


def achar_label(img: Path) -> Path | None:
    candidatos = [img.with_suffix(".txt")]
    partes = list(img.parts)
    for i in range(len(partes) - 1, -1, -1):
        if partes[i].lower() in ("images", "imagens", "imgs"):
            novo = partes.copy()
            novo[i] = "labels"
            candidatos.append(Path(*novo).with_suffix(".txt"))
            break
    return next((c for c in candidatos if c.exists()), None)


def split_original(img: Path) -> str | None:
    for parte in img.relative_to(ORIGEM).parts[:-1]:
        p = parte.lower()
        if p in ("train", "treino", "training"):
            return "train"
        if p in ("val", "valid", "validation", "validacao", "validação"):
            return "val"
        if p in ("test", "teste", "testing"):
            return "test"
    return None


def converter_linha(linha: str) -> tuple[int, list[float]] | None:
    v = linha.split()
    if len(v) < 5:
        return None
    try:
        cls = int(float(v[0]))
        nums = [float(x) for x in v[1:]]
    except ValueError:
        return None
    if len(nums) == 4:
        xc, yc, w, h = nums
    elif len(nums) >= 6 and len(nums) % 2 == 0:  # polígono / OBB
        xs, ys = nums[0::2], nums[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    else:
        return None
    # recorta para dentro da imagem
    x1, y1 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
    x2, y2 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
    if x2 - x1 <= 0.001 or y2 - y1 <= 0.001:
        return None
    return cls, [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def grupo_da_imagem(img: Path, cache: dict) -> str:
    """Fotos do mesmo hidrômetro (mesma leitura sem os 3 últimos dígitos, segundo
    o anotacoes.json do rotular.py) formam um grupo e não são separadas entre
    treino e validação -> a precisão medida fica honesta."""
    pasta = img.parent.parent if img.parent.name == "images" else img.parent
    if pasta not in cache:
        arq = pasta / "anotacoes.json"
        cache[pasta] = json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else {}
    leitura = cache[pasta].get(img.name, {}).get("leitura")
    return f"{pasta}|{leitura[:-3]}" if leitura and len(leitura) > 4 else str(img)


def _caixas_px(linhas: list[str], w: int, h: int):
    out = []
    for l in linhas:
        c, xc, yc, bw, bh = l.split()
        c, xc, yc, bw, bh = int(c), float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
        out.append((c, int(round(xc - bw / 2)), int(round(yc - bh / 2)),
                    int(round(xc + bw / 2)), int(round(yc + bh / 2))))
    return out


def gerar_sinteticas(treino: list[tuple[Path, list[str]]], n_por_imagem: int,
                     destino_img: Path, destino_lbl: Path) -> Counter:
    """Cria variações das imagens de treino trocando o conteúdo de cada caixa de
    dígito por outro dígito (da mesma foto quando possível, para manter o estilo).
    Dígitos raros são sorteados com mais frequência."""
    rng = random.Random(SEMENTE)
    fontes = []                          # (img, caixas de dígito)
    banco = defaultdict(list)            # classe -> recortes (de todas as fotos)
    for caminho, linhas in treino:
        img = ler_imagem(caminho)
        if img is None:
            continue
        H, W = img.shape[:2]
        digs = [b for b in _caixas_px(linhas, W, H) if b[0] < 10 and b[3] - b[1] > 2 and b[4] - b[2] > 2]
        if len(digs) < 3:
            continue
        fontes.append((caminho, img, linhas, digs))
        for c, x1, y1, x2, y2 in digs:
            if len(banco[c]) < 400:
                banco[c].append(img[max(0, y1):y2, max(0, x1):x2].copy())
    if not fontes:
        return Counter()

    freq = Counter(c for *_, digs in fontes for c, *_ in digs)
    pesos = np.array([1.0 / (freq[c] + 5) if banco[c] else 0.0 for c in range(10)])
    pesos /= pesos.sum()

    contagem = Counter()
    for caminho, img, linhas, digs in fontes:
        H, W = img.shape[:2]
        mesma_foto = defaultdict(list)
        for c, x1, y1, x2, y2 in digs:
            mesma_foto[c].append(img[max(0, y1):y2, max(0, x1):x2].copy())
        for v in range(n_por_imagem):
            nova = img.copy()
            novas_linhas = [l for l in linhas if int(l.split()[0]) >= 10]   # mantém o visor
            for c, x1, y1, x2, y2 in digs:
                nc = int(rng.choices(range(10), weights=pesos)[0])
                opcoes = mesma_foto[nc] if mesma_foto[nc] and rng.random() < 0.7 else banco[nc]
                tile = opcoes[rng.randrange(len(opcoes))]
                x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                nova[y1c:y2c, x1c:x2c] = cv2.resize(tile, (x2c - x1c, y2c - y1c))
                novas_linhas.append(f"{nc} {(x1c + x2c) / 2 / W:.6f} {(y1c + y2c) / 2 / H:.6f} "
                                    f"{(x2c - x1c) / W:.6f} {(y2c - y1c) / H:.6f}")
                contagem[nc] += 1
            nome = f"{caminho.stem}_sint{v:02d}"
            salvar_imagem(destino_img / f"{nome}.jpg", nova)
            (destino_lbl / f"{nome}.txt").write_text("\n".join(novas_linhas), encoding="utf-8")
    return contagem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sinteticas", type=int, default=10,
                    help="imagens sintéticas por foto de treino (0 = desliga)")
    args = ap.parse_args()

    if not ORIGEM.exists():
        sys.exit(f"Pasta {ORIGEM} não existe.")
    imagens = sorted(p for p in ORIGEM.rglob("*") if p.suffix.lower() in EXTENSOES_IMAGEM)
    if not imagens:
        sys.exit(f"Nenhuma imagem encontrada em {ORIGEM}. Copie seus datasets para lá.")

    print(f"Encontradas {len(imagens)} imagens em {ORIGEM}")
    cache_nomes: dict[Path, dict] = {}
    avisos_sem_nomes = set()
    ignoradas = Counter()
    contagem = Counter()
    itens = []  # (img, linhas_convertidas, split)
    sem_label = 0
    sem_digitos: dict[Path, list] = {}

    for img in imagens:
        lbl = achar_label(img)
        if lbl is None:
            sem_label += 1
            continue
        pasta = lbl.parent
        if pasta not in cache_nomes:
            nomes = achar_nomes_classes(pasta)
            if nomes is None:
                nomes = {i: str(i) for i in range(10)}
                avisos_sem_nomes.add(pasta)
            cache_nomes[pasta] = nomes
        nomes = cache_nomes[pasta]
        mapa_dig, ids_visor = mapear_classes(nomes)
        if not mapa_dig:
            # dataset sem classes de dígito (ex.: só "hidrometro"): não ensina a
            # ler e, se entrasse, viraria "foto sem dígitos" e atrapalharia o treino
            sem_digitos[pasta.parent] = list(nomes.values())
            continue

        linhas, n_linhas = [], 0
        for linha in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            conv = converter_linha(linha)
            if conv is None:
                continue
            n_linhas += 1
            cls, caixa = conv
            if cls in mapa_dig:
                novo = mapa_dig[cls]
            elif cls in ids_visor:
                novo = ID_VISOR
            else:
                ignoradas[nomes.get(cls, str(cls))] += 1
                continue
            linhas.append(f"{novo} " + " ".join(f"{v:.6f}" for v in caixa))
            contagem[novo] += 1
        if n_linhas and not linhas:
            continue   # tinha rótulos, mas nenhum útil: não usar como foto "vazia"
        itens.append((img, linhas, split_original(img)))

    for p, nms in sem_digitos.items():
        print(f"\nIGNORADO: {p}\n  classes {nms} não são dígitos 0-9. As caixas podem ser "
              f"aproveitadas com:\n  python importar_roboflow.py  e depois  python rotular.py")

    if not itens:
        sys.exit("\nNenhuma imagem com rótulo de dígito encontrada. Anote com:  python rotular.py")

    # divisão train/val/test
    random.seed(SEMENTE)
    tem_split = any(s for _, _, s in itens)
    grupos = defaultdict(list)          # fotos do mesmo hidrômetro juntas
    cache_anot = {}
    for i, (img, _, s) in enumerate(itens):
        if s is None:
            grupos[grupo_da_imagem(img, cache_anot)].append(i)
    lista_grupos = list(grupos.values())
    random.shuffle(lista_grupos)
    n = sum(len(g) for g in lista_grupos)
    acumulado = 0
    for g in lista_grupos:
        frac = acumulado / max(1, n)
        s = "train" if frac < 0.80 else ("val" if frac < 0.95 else "test")
        for idx in g:
            itens[idx] = (itens[idx][0], itens[idx][1], s)
        acumulado += len(g)
    # garante que exista validação
    if not any(s == "val" for _, _, s in itens):
        treino = [i for i, it in enumerate(itens) if it[2] == "train"]
        random.shuffle(treino)
        for idx in treino[: max(1, len(treino) // 7)]:
            itens[idx] = (itens[idx][0], itens[idx][1], "val")

    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    por_split = Counter()
    usados = set()
    treino_copiado = []
    for k, (img, linhas, s) in enumerate(itens):
        nome = f"{k:06d}_{img.stem[:60]}"
        if nome in usados:
            continue
        usados.add(nome)
        (DESTINO / "images" / s).mkdir(parents=True, exist_ok=True)
        (DESTINO / "labels" / s).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, DESTINO / "images" / s / f"{nome}{img.suffix.lower()}")
        (DESTINO / "labels" / s / f"{nome}.txt").write_text("\n".join(linhas), encoding="utf-8")
        por_split[s] += 1
        if s == "train":
            treino_copiado.append((DESTINO / "images" / s / f"{nome}{img.suffix.lower()}", linhas))

    cont_sint = Counter()
    if args.sinteticas > 0 and treino_copiado:
        print(f"\nGerando {args.sinteticas} imagens sintéticas por foto de treino...")
        cont_sint = gerar_sinteticas(treino_copiado, args.sinteticas,
                                     DESTINO / "images" / "train", DESTINO / "labels" / "train")
        por_split["sinteticas"] = sum(1 for _ in (DESTINO / "images" / "train").glob("*_sint*"))

    tem_visor = contagem[ID_VISOR] > 0
    nomes_finais = CLASSES_FINAIS if tem_visor else CLASSES_FINAIS[:10]
    data = {
        "path": str(DESTINO),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(nomes_finais)),
    }
    if por_split["test"]:
        data["test"] = "images/test"
    with open(DESTINO / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    # relatório
    print("\n=========== RESUMO ===========")
    print(f"Imagens reais: {por_split['train'] + por_split['val'] + por_split['test']}  "
          f"(train={por_split['train']}, val={por_split['val']}, test={por_split['test']})")
    if por_split["sinteticas"]:
        print(f"Imagens sintéticas no treino: {por_split['sinteticas']}  "
              f"(+{sum(cont_sint.values())} dígitos: "
              + ", ".join(f"{c}={cont_sint[c]}" for c in range(10)) + ")")
    print(f"Divisão original preservada: {'sim' if tem_split else 'não (dividido 80/15/5)'}")
    if sem_label:
        print(f"Imagens sem rótulo ignoradas: {sem_label}")
    print("\nObjetos por classe:")
    for c in range(len(nomes_finais)):
        barra = "#" * min(50, contagem[c] // max(1, max(contagem.values()) // 50))
        print(f"  {nomes_finais[c]:>6}: {contagem[c]:6d} {barra}")
    if ignoradas:
        print(f"\nClasses ignoradas (não são dígito nem visor): {dict(ignoradas)}")
    for p in avisos_sem_nomes:
        print(f"\nAVISO: sem data.yaml/classes.txt em {p} - assumindo classes 0..9 = dígitos")
    faltando = [c for c in range(10) if contagem[c] == 0]
    if faltando:
        print(f"\nAVISO: dígitos sem nenhum exemplo: {faltando} - o modelo não aprenderá esses!")
    poucos = [c for c in range(10) if 0 < contagem[c] < 100]
    if poucos:
        print(f"AVISO: dígitos com poucos exemplos (<100): {poucos} - considere mais imagens")
    print(f"\nPronto! Arquivo gerado: {DESTINO / 'data.yaml'}")
    print("Próximo passo:  python treinar.py")


if __name__ == "__main__":
    main()
