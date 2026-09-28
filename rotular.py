"""
Ferramenta de anotação rápida das fotos de hidrômetro.

Para cada foto em  imagens_treino/ :
  1. (opcional) arraste um retângulo com o mouse para dar ZOOM no visor
     (ou use a rodinha do mouse)
  2. clique nos 4 CANTOS da janela de dígitos (qualquer ordem)
     -> contorne só os dígitos, sem o "m³"
  3. digite a leitura completa (ex.: 03606911) e tecle ENTER

A ferramenta divide a janela em uma caixa por dígito e grava os rótulos
YOLO em  dataset/proprio/ . Pode fechar a qualquer momento: ao abrir de novo,
ela continua da primeira foto ainda não anotada.

Dígito "rolando" (entre dois números): digite o MENOR (ex.: entre 3 e 4 -> 3;
entre 9 e 0 -> 9). Foto ilegível ou sem hidrômetro: botão "Descartar".

Uso:  python rotular.py
"""
from __future__ import annotations

import json
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np
import yaml
from PIL import Image, ImageTk

from leitor import EXTENSOES_IMAGEM, ler_imagem

RAIZ = Path(__file__).resolve().parent
ORIGEM = RAIZ / "imagens_treino"
DESTINO = RAIZ / "dataset" / "proprio"
ARQ_ANOT = DESTINO / "anotacoes.json"
NOMES = [str(i) for i in range(10)] + ["visor"]
ID_VISOR = 10


# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------
def ordenar_cantos(pts) -> np.ndarray:
    """Retorna [sup-esq, sup-dir, inf-dir, inf-esq]."""
    p = np.array(pts, dtype=np.float64)
    s, d = p.sum(1), p[:, 1] - p[:, 0]
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]])


def quads_digitos(pts, n: int) -> list[np.ndarray]:
    """Divide o quadrilátero do visor em n quadriláteros (um por dígito)."""
    tl, tr, br, bl = ordenar_cantos(pts)
    quads = []
    for i in range(n):
        a, b = i / n, (i + 1) / n
        quads.append(np.array([tl + (tr - tl) * a, tl + (tr - tl) * b,
                               bl + (br - bl) * b, bl + (br - bl) * a]))
    return quads


def gerar_labels(pts, leitura: str, w: int, h: int) -> list[str]:
    linhas = []

    def caixa(q, cls, margem=0.0):
        x1, y1 = q.min(0)
        x2, y2 = q.max(0)
        mx, my = (x2 - x1) * margem, (y2 - y1) * margem
        x1, x2 = max(0, x1 - mx), min(w, x2 + mx)
        y1, y2 = max(0, y1 - my), min(h, y2 + my)
        linhas.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                      f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")

    for q, ch in zip(quads_digitos(pts, len(leitura)), leitura):
        caixa(q, int(ch))
    caixa(ordenar_cantos(pts), ID_VISOR, margem=0.04)
    return linhas


def gerar_labels_caixas(caixas, leitura: str, w: int, h: int) -> list[str]:
    """Rótulos a partir de caixas individuais de dígito (ex.: importadas)."""
    linhas = []
    for (x1, y1, x2, y2), ch in zip(caixas, leitura):
        linhas.append(f"{ch} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                      f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    a = np.array(caixas)
    x1, y1, x2, y2 = a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()
    mx, my = (x2 - x1) * 0.04, (y2 - y1) * 0.04
    x1, y1, x2, y2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    linhas.append(f"{ID_VISOR} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                  f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return linhas


def concluida(a: dict) -> bool:
    return bool(a.get("leitura") or a.get("descartada"))


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class App:
    def __init__(self, raiz: tk.Tk):
        self.raiz = raiz
        self.imagens = sorted(p for p in ORIGEM.iterdir() if p.suffix.lower() in EXTENSOES_IMAGEM)
        if not self.imagens:
            messagebox.showerror("Sem imagens", f"Coloque as fotos em {ORIGEM}")
            raiz.destroy()
            return
        DESTINO.joinpath("images").mkdir(parents=True, exist_ok=True)
        DESTINO.joinpath("labels").mkdir(parents=True, exist_ok=True)
        with open(DESTINO / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"names": NOMES}, f)
        self.anot = json.loads(ARQ_ANOT.read_text(encoding="utf-8")) if ARQ_ANOT.exists() else {}

        sw, sh = raiz.winfo_screenwidth(), raiz.winfo_screenheight()
        self.CW, self.CH = min(1400, sw - 80), min(850, sh - 220)

        raiz.title("Anotação de hidrômetros")
        self.status = tk.Label(raiz, font=("Segoe UI", 11, "bold"), anchor="w")
        self.status.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(raiz, anchor="w", fg="#555", font=("Segoe UI", 9),
                 text="Arraste = zoom  |  rodinha = zoom  |  clique = canto (4 cantos da janela de dígitos)  |  "
                      "digite a leitura + ENTER  |  ESC = limpar cantos  |  dígito rolando: digite o MENOR"
                 ).pack(fill="x", padx=8)

        self.canvas = tk.Canvas(raiz, width=self.CW, height=self.CH, bg="#222", cursor="crosshair")
        self.canvas.pack(padx=8, pady=6)

        barra = tk.Frame(raiz)
        barra.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(barra, text="Leitura:", font=("Segoe UI", 12)).pack(side="left")
        self.var_leitura = tk.StringVar()
        self.var_leitura.trace_add("write", lambda *_: self.redesenhar())
        self.entrada = tk.Entry(barra, textvariable=self.var_leitura, font=("Consolas", 16), width=14)
        self.entrada.pack(side="left", padx=6)
        for texto, cmd in [("Salvar e próxima (Enter)", self.salvar),
                           ("Limpar cantos (Esc)", self.limpar),
                           ("Ver foto inteira", self.ver_inteira),
                           ("◀ Anterior (PgUp)", lambda: self.ir(self.idx - 1)),
                           ("Pular ▶ (PgDn)", lambda: self.ir(self.idx + 1)),
                           ("Descartar foto", self.descartar)]:
            tk.Button(barra, text=texto, command=cmd).pack(side="left", padx=3)

        self.canvas.bind("<ButtonPress-1>", self.ao_pressionar)
        self.canvas.bind("<B1-Motion>", self.ao_arrastar)
        self.canvas.bind("<ButtonRelease-1>", self.ao_soltar)
        self.canvas.bind("<MouseWheel>", self.ao_rodinha)
        raiz.bind("<Return>", lambda e: self.salvar())
        raiz.bind("<Escape>", lambda e: self.limpar())
        raiz.bind("<Prior>", lambda e: self.ir(self.idx - 1))
        raiz.bind("<Next>", lambda e: self.ir(self.idx + 1))

        pend = [i for i, p in enumerate(self.imagens) if not concluida(self.anot.get(p.name, {}))]
        self.ir(pend[0] if pend else 0)

    # -- navegação -------------------------------------------------------------
    def ir(self, idx: int):
        if not 0 <= idx < len(self.imagens):
            if idx >= len(self.imagens):
                messagebox.showinfo("Fim", "Última foto. Rode agora:  python preparar_dataset.py")
            return
        self.idx = idx
        self.caminho = self.imagens[idx]
        bgr = ler_imagem(self.caminho)
        if bgr is None:
            messagebox.showwarning("Foto inválida", f"Não consegui abrir {self.caminho.name}. Pulando.")
            self.ir(idx + 1)
            return
        self.img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.W, self.H = self.img.size
        a = self.anot.get(self.caminho.name, {})
        self.pts = [tuple(p) for p in a.get("pontos", [])]
        self.caixas = [tuple(b) for b in a.get("caixas", [])]
        self.var_leitura.set(a.get("leitura", ""))
        self.drag = None
        if len(self.pts) == 4:   # já anotada: abre com zoom no visor
            q = np.array(self.pts)
            (x1, y1), (x2, y2) = q.min(0), q.max(0)
            m = max(x2 - x1, y2 - y1) * 0.6
            self.zoom_para(x1 - m, y1 - m, x2 + m, y2 + m)
        else:
            self.ver_inteira()
        self.entrada.focus_set()
        self.entrada.icursor("end")

    def ver_inteira(self):
        self.zoom_para(0, 0, self.W, self.H)

    def zoom_para(self, x1, y1, x2, y2):
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(self.W), x2), min(float(self.H), y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return
        self.vista = (x1, y1, x2 - x1, y2 - y1)
        vx, vy, vw, vh = self.vista
        self.esc = min(self.CW / vw, self.CH / vh)
        self.ox = (self.CW - vw * self.esc) / 2
        self.oy = (self.CH - vh * self.esc) / 2
        recorte = self.img.crop((int(vx), int(vy), int(vx + vw), int(vy + vh)))
        tam = (max(1, int(vw * self.esc)), max(1, int(vh * self.esc)))
        filtro = Image.Resampling.LANCZOS if self.esc < 1 else Image.Resampling.BICUBIC
        self.foto_tk = ImageTk.PhotoImage(recorte.resize(tam, filtro))
        self.redesenhar()

    # -- coordenadas -----------------------------------------------------------
    def p_canvas(self, x, y):
        vx, vy, _, _ = self.vista
        return (x - vx) * self.esc + self.ox, (y - vy) * self.esc + self.oy

    def p_imagem(self, cx, cy):
        vx, vy, _, _ = self.vista
        return (cx - self.ox) / self.esc + vx, (cy - self.oy) / self.esc + vy

    # -- mouse -----------------------------------------------------------------
    def ao_pressionar(self, e):
        self.drag = (e.x, e.y, e.x, e.y)

    def ao_arrastar(self, e):
        if self.drag:
            self.drag = (*self.drag[:2], e.x, e.y)
            self.redesenhar()

    def ao_soltar(self, e):
        if not self.drag:
            return
        x0, y0, _, _ = self.drag
        self.drag = None
        if abs(e.x - x0) > 8 and abs(e.y - y0) > 8:          # arrastou -> zoom
            a, b = self.p_imagem(min(x0, e.x), min(y0, e.y))
            c, d = self.p_imagem(max(x0, e.x), max(y0, e.y))
            self.zoom_para(a, b, c, d)
        else:                                                   # clique -> canto
            if len(self.pts) >= 4:
                self.pts = []
                self.caixas = []     # clicar cantos substitui as caixas importadas
            x, y = self.p_imagem(e.x, e.y)
            self.pts.append((min(max(x, 0), self.W), min(max(y, 0), self.H)))
            self.redesenhar()
        self.entrada.focus_set()

    def ao_rodinha(self, e):
        f = 0.8 if e.delta > 0 else 1.25
        cx, cy = self.p_imagem(e.x, e.y)
        vx, vy, vw, vh = self.vista
        nw, nh = vw * f, vh * f
        rx, ry = (cx - vx) / vw, (cy - vy) / vh
        self.zoom_para(cx - rx * nw, cy - ry * nh, cx - rx * nw + nw, cy - ry * nh + nh)

    # -- desenho ---------------------------------------------------------------
    def redesenhar(self):
        if not hasattr(self, "foto_tk"):
            return
        c = self.canvas
        c.delete("all")
        c.create_image(self.ox, self.oy, image=self.foto_tk, anchor="nw")
        for p in self.pts:
            x, y = self.p_canvas(*p)
            c.create_oval(x - 5, y - 5, x + 5, y + 5, outline="#ff0", width=2)
        leitura = "".join(ch for ch in self.var_leitura.get() if ch.isdigit())
        if self.caixas:
            for k, (x1, y1, x2, y2) in enumerate(self.caixas):
                a, b = self.p_canvas(x1, y1)
                cc, d = self.p_canvas(x2, y2)
                c.create_rectangle(a, b, cc, d, outline="#0f0", width=2)
                if k < len(leitura):
                    c.create_text((a + cc) / 2, b - 12, text=leitura[k], fill="#0ff",
                                  font=("Consolas", 14, "bold"))
        elif len(self.pts) == 4:
            q = ordenar_cantos(self.pts)
            c.create_polygon(*[v for p in q for v in self.p_canvas(*p)],
                             outline="#0f0", fill="", width=2)
            if leitura:
                for dq, ch in zip(quads_digitos(self.pts, len(leitura)), leitura):
                    pts_c = [self.p_canvas(*p) for p in dq]
                    c.create_line(*pts_c[1], *pts_c[2], fill="#0ff", width=1)
                    cx = sum(p[0] for p in pts_c) / 4
                    y_top = min(p[1] for p in pts_c)
                    c.create_text(cx, y_top - 12, text=ch, fill="#0ff",
                                  font=("Consolas", 14, "bold"))
        if self.drag:
            c.create_rectangle(*self.drag, outline="#f80", dash=(4, 3), width=2)
        feitas = sum(1 for p in self.imagens if concluida(self.anot.get(p.name, {})))
        a = self.anot.get(self.caminho.name, {})
        estado = "DESCARTADA" if a.get("descartada") else ("anotada" if a.get("leitura") else "pendente")
        if self.caixas:
            marca = f"{len(self.caixas)} caixas prontas -> digite {len(self.caixas)} dígitos (ESC = refazer com 4 cantos)"
        else:
            marca = f"cantos: {len(self.pts)}/4" + ("  (pré-marcados - confira)" if a.get("importada") and self.pts else "")
        self.status.config(text=f"Foto {self.idx + 1}/{len(self.imagens)}  -  {self.caminho.name}  "
                                f"[{estado}]   |   concluídas: {feitas}/{len(self.imagens)}   |   {marca}")

    # -- ações -----------------------------------------------------------------
    def limpar(self):
        self.pts = []
        self.caixas = []
        self.redesenhar()

    def _gravar_json(self):
        ARQ_ANOT.write_text(json.dumps(self.anot, indent=1, ensure_ascii=False), encoding="utf-8")

    def _remover_saida(self):
        (DESTINO / "labels" / f"{self.caminho.stem}.txt").unlink(missing_ok=True)
        (DESTINO / "images" / self.caminho.name).unlink(missing_ok=True)

    def salvar(self):
        leitura = "".join(ch for ch in self.var_leitura.get() if ch.isdigit())
        if len(self.pts) != 4 and not self.caixas:
            messagebox.showwarning("Faltam cantos", "Clique nos 4 cantos da janela de dígitos.")
            return
        if len(leitura) < 3:
            messagebox.showwarning("Leitura", "Digite a leitura completa (todos os dígitos).")
            return
        if self.caixas and len(leitura) != len(self.caixas):
            messagebox.showwarning(
                "Quantidade", f"Esta foto tem {len(self.caixas)} caixas de dígito e você digitou "
                              f"{len(leitura)} dígitos.\nCorrija a leitura, ou tecle ESC e clique "
                              f"os 4 cantos para refazer.")
            return
        if self.caixas:
            linhas = gerar_labels_caixas(self.caixas, leitura, self.W, self.H)
        else:
            linhas = gerar_labels(self.pts, leitura, self.W, self.H)
        shutil.copy2(self.caminho, DESTINO / "images" / self.caminho.name)
        (DESTINO / "labels" / f"{self.caminho.stem}.txt").write_text("\n".join(linhas), encoding="utf-8")
        self.anot[self.caminho.name] = {"leitura": leitura,
                                        "pontos": [list(map(float, p)) for p in self.pts]}
        if self.caixas:
            self.anot[self.caminho.name]["caixas"] = [list(map(float, b)) for b in self.caixas]
        self._gravar_json()
        self.ir(self.idx + 1)

    def descartar(self):
        self._remover_saida()
        self.anot[self.caminho.name] = {"descartada": True}
        self._gravar_json()
        self.ir(self.idx + 1)


if __name__ == "__main__":
    raiz = tk.Tk()
    App(raiz)
    raiz.mainloop()
