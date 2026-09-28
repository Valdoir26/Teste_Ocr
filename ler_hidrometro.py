"""
Lê o consumo dos hidrômetros.

Coloque as fotos na pasta  imagens_entrada/  e rode:
  python ler_hidrometro.py

Ou leia uma imagem específica:
  python ler_hidrometro.py --imagem caminho/da/foto.jpg

Resultados em  resultados/ :
  - leituras.csv          (uma linha por imagem)
  - <nome>_leitura.jpg    (imagem com os dígitos marcados)
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from leitor import LeitorHidrometro, carregar_config, ler_imagem, listar_imagens, salvar_imagem

RAIZ = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagem", help="lê só esta imagem")
    ap.add_argument("--pasta", help="pasta de entrada (padrão: config.yaml)")
    ap.add_argument("--modelo", help="caminho do modelo .pt")
    ap.add_argument("--conf", type=float, help="confiança mínima")
    args = ap.parse_args()

    cfg = carregar_config(RAIZ / "config.yaml")["leitura"]
    if args.modelo:
        cfg["modelo"] = args.modelo
    if args.conf is not None:
        cfg["confianca_minima"] = args.conf
    cfg["modelo"] = str((RAIZ / cfg["modelo"]).resolve()) if not Path(cfg["modelo"]).is_absolute() else cfg["modelo"]

    if args.imagem:
        imagens = [Path(args.imagem)]
    else:
        pasta = Path(args.pasta) if args.pasta else RAIZ / cfg["pasta_entrada"]
        imagens = listar_imagens(pasta)
        if not imagens:
            sys.exit(f"Nenhuma imagem em {pasta}. Coloque as fotos dos hidrômetros lá.")

    saida = RAIZ / cfg["pasta_saida"]
    saida.mkdir(exist_ok=True)

    leitor = LeitorHidrometro(cfg)
    print(f"Modelo: {cfg['modelo']}")
    print(f"Lendo {len(imagens)} imagem(ns)...\n")
    print(f"{'ARQUIVO':40} {'LEITURA':>14} {'CONF':>6}  AVISOS")
    print("-" * 90)

    linhas = []
    for caminho in imagens:
        img = ler_imagem(caminho)
        if img is None:
            print(f"{caminho.name[:40]:40} {'ERRO':>14}         não foi possível abrir")
            continue
        res = leitor.ler(img, caminho.name)
        aviso = "; ".join(res.avisos)
        print(f"{caminho.name[:40]:40} {res.leitura or '---':>14} {res.confianca_media:6.0%}  {aviso}")

        if cfg.get("salvar_imagem_anotada", True):
            salvar_imagem(saida / f"{caminho.stem}_leitura.jpg", leitor.desenhar(img, res))

        linhas.append({
            "arquivo": caminho.name,
            "leitura": res.leitura,
            "digitos": res.digitos_str,
            "valor_m3": "" if res.valor_m3 is None else f"{res.valor_m3:.3f}",
            "confianca_media": f"{res.confianca_media:.3f}",
            "confianca_minima": f"{res.confianca_minima:.3f}",
            "completa": "sim" if res.completa else "nao",
            "conferir": "sim" if (not res.completa or res.confianca_minima < 0.5) else "nao",
            "avisos": aviso,
            "data_leitura": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    if linhas:
        csv_path = saida / "leituras.csv"
        # utf-8-sig + ';' para abrir direto no Excel em português
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()), delimiter=";")
            w.writeheader()
            w.writerows(linhas)
        conferir = sum(1 for l in linhas if l["conferir"] == "sim")
        print("-" * 90)
        print(f"\n{len(linhas)} leitura(s) salvas em {csv_path}")
        if conferir:
            print(f"{conferir} leitura(s) marcadas para conferência manual (coluna 'conferir').")


if __name__ == "__main__":
    main()
