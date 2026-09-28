"""
Treina o YOLO para detectar os dígitos do hidrômetro.

Uso:
  python treinar.py                     # usa config.yaml
  python treinar.py --modelo yolo26m.pt # troca o modelo base
  python treinar.py --continuar         # retoma treino interrompido
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

from leitor import carregar_config

RAIZ = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", help="modelo base (ex.: yolo26s.pt, yolo26m.pt)")
    ap.add_argument("--epocas", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=float)
    ap.add_argument("--continuar", action="store_true", help="retoma o último treino")
    args = ap.parse_args()

    cfg = carregar_config(RAIZ / "config.yaml")["treino"]
    data = RAIZ / "dataset_preparado" / "data.yaml"
    if not data.exists():
        sys.exit("dataset_preparado/data.yaml não existe. Rode antes:  python preparar_dataset.py")

    from ultralytics import YOLO

    projeto = RAIZ / "runs"
    nome = cfg["nome_experimento"]
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {torch.cuda.get_device_name(0) if dispositivo == 0 else 'CPU (vai ser lento!)'}")

    if args.continuar:
        ultimo = projeto / nome / "weights" / "last.pt"
        if not ultimo.exists():
            sys.exit(f"Nada para continuar: {ultimo} não existe")
        resultado = YOLO(str(ultimo)).train(resume=True)
    else:
        batch = args.batch if args.batch is not None else cfg["batch"]
        if batch is not None and (batch >= 1 or batch == -1):
            batch = int(batch)
        modelo = YOLO(args.modelo or cfg["modelo_base"])
        resultado = modelo.train(
            data=str(data),
            epochs=args.epocas or cfg["epocas"],
            patience=cfg["paciencia"],
            imgsz=args.imgsz or cfg["imgsz"],
            batch=batch,
            workers=cfg["workers"],
            device=dispositivo,
            project=str(projeto),
            name=nome,
            exist_ok=True,
            seed=42,
            cos_lr=True,
            close_mosaic=15,
            cache=False,
            plots=True,
            # ---- aumentos de dados apropriados para dígitos ----
            fliplr=0.0,        # NUNCA espelhar: 2<->5, 6<->9 ficariam confusos
            flipud=0.0,
            degrees=7.0,       # fotos levemente tortas
            shear=2.0,
            perspective=0.0005,
            translate=0.1,
            scale=0.5,
            mosaic=1.0,
            mixup=0.05,
            hsv_h=0.02,        # hidrômetros têm iluminação/cores variadas
            hsv_s=0.6,
            hsv_v=0.5,
        )

    # copia o melhor modelo para modelos/
    save_dir = getattr(resultado, "save_dir", None)
    melhor = Path(save_dir) / "weights" / "best.pt" if save_dir else \
        projeto / nome / "weights" / "best.pt"
    if melhor.exists():
        destino = RAIZ / "modelos" / "hidrometro_best.pt"
        destino.parent.mkdir(exist_ok=True)
        shutil.copy2(melhor, destino)
        print(f"\nMelhor modelo salvo em: {destino}")
        print("Gráficos e métricas em:", melhor.parent.parent)
        print("\nPróximos passos:")
        print("  python avaliar.py          # mede a precisão da leitura completa")
        print("  python ler_hidrometro.py   # lê as imagens de imagens_entrada/")


if __name__ == "__main__":
    main()
