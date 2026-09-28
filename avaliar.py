"""
Avalia a precisão do modelo nas imagens de teste (ou validação).

Mede:
  * mAP das detecções (métrica padrão YOLO)
  * % de leituras 100% corretas (todos os dígitos certos, na ordem)
  * % de dígitos corretos

Os erros são salvos em resultados/erros_avaliacao/ para você analisar.

Uso:  python avaliar.py [--split val|test]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from leitor import (Digito, LeitorHidrometro, carregar_config, ler_imagem,
                    listar_imagens, ordenar, salvar_imagem)

RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "dataset_preparado"


def leitura_verdadeira(label: Path, w: int, h: int) -> str:
    digs = []
    for linha in label.read_text(encoding="utf-8").splitlines():
        v = linha.split()
        if len(v) != 5:
            continue
        c, xc, yc, bw, bh = int(v[0]), *map(float, v[1:])
        if c > 9:
            continue
        digs.append(Digito((xc - bw / 2) * w, (yc - bh / 2) * h,
                           (xc + bw / 2) * w, (yc + bh / 2) * h, c, 1.0))
    return "".join(str(d.valor) for d in ordenar(digs))


def distancia_edicao(a: str, b: str) -> int:
    ant = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        atual = [i]
        for j, cb in enumerate(b, 1):
            atual.append(min(ant[j] + 1, atual[j - 1] + 1, ant[j - 1] + (ca != cb)))
        ant = atual
    return ant[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, help="val ou test (padrão: test se existir)")
    ap.add_argument("--modelo")
    args = ap.parse_args()

    cfg = carregar_config(RAIZ / "config.yaml")["leitura"]
    if args.modelo:
        cfg["modelo"] = args.modelo
    if not Path(cfg["modelo"]).is_absolute():
        cfg["modelo"] = str(RAIZ / cfg["modelo"])

    split = args.split or ("test" if (DADOS / "images" / "test").exists() else "val")
    imagens = listar_imagens(DADOS / "images" / split)
    if not imagens:
        sys.exit(f"Sem imagens em {DADOS / 'images' / split}. Rode preparar_dataset.py")

    # 1) métricas de detecção padrão
    from ultralytics import YOLO
    print(f"=== mAP no split '{split}' ===")
    m = YOLO(cfg["modelo"]).val(data=str(DADOS / "data.yaml"), split=split,
                                imgsz=cfg["imgsz"], plots=False, verbose=False)
    print(f"mAP50: {m.box.map50:.4f}   mAP50-95: {m.box.map:.4f}   "
          f"Precisão: {m.box.mp:.4f}   Recall: {m.box.mr:.4f}")

    # 2) precisão da leitura completa
    print(f"\n=== Leitura completa em {len(imagens)} imagens ===")
    leitor = LeitorHidrometro(cfg)
    pasta_erros = RAIZ / "resultados" / "erros_avaliacao"
    pasta_erros.mkdir(parents=True, exist_ok=True)

    total = certas = dig_total = dig_erros = 0
    erros = []
    for p in imagens:
        label = DADOS / "labels" / split / f"{p.stem}.txt"
        img = ler_imagem(p)
        if img is None or not label.exists():
            continue
        esperado = leitura_verdadeira(label, img.shape[1], img.shape[0])
        if not esperado:
            continue
        res = leitor.ler(img, p.name)
        lido = res.digitos_str
        total += 1
        dig_total += len(esperado)
        d = distancia_edicao(esperado, lido)
        dig_erros += d
        if lido == esperado:
            certas += 1
        else:
            erros.append({"arquivo": p.name, "esperado": esperado, "lido": lido,
                          "confianca": f"{res.confianca_media:.3f}",
                          "avisos": "; ".join(res.avisos)})
            lido_arq = (lido or "vazio").replace("?", "X")  # '?' é inválido em nome de arquivo
            salvar_imagem(pasta_erros / f"{p.stem}_esp{esperado}_lido{lido_arq}.jpg",
                          leitor.desenhar(img, res))

    if not total:
        sys.exit("Nenhuma imagem com dígitos rotulados para avaliar.")

    print(f"Leituras 100% corretas: {certas}/{total} = {certas / total:.2%}")
    print(f"Dígitos corretos:       {1 - dig_erros / dig_total:.2%}")
    if erros:
        with open(pasta_erros / "erros.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(erros[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(erros)
        print(f"\n{len(erros)} erro(s) salvos em {pasta_erros}")
        for e in erros[:15]:
            print(f"  {e['arquivo'][:40]:40} esperado={e['esperado']:>10}  lido={e['lido']:>10}")


if __name__ == "__main__":
    main()
