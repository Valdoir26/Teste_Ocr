"""
Núcleo da leitura de hidrômetros.

Fluxo:
  1. YOLO detecta dígitos (classes 0-9) e, se existir, o visor/contador.
  2. Se houver visor: recorta, e roda o YOLO de novo no recorte (dígitos
     ficam maiores -> mais precisão). Fica com a melhor das duas leituras.
  3. Remove caixas duplicadas no mesmo lugar (trata dígito "rolando").
  4. Agrupa os dígitos que estão na mesma linha (descarta nº de série e
     outros números impressos no hidrômetro).
  5. Ordena da esquerda p/ direita seguindo a inclinação da linha
     (funciona com foto torta).
  6. Detecta dígitos faltando pelo espaçamento e monta a leitura.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch
    from ultralytics.engine.results import Results

EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Nomes de classe reconhecidos como "visor/contador"
PALAVRAS_VISOR = ("display", "visor", "counter", "contador", "reading", "leitura",
                  "meter", "medidor", "odometer", "roi", "screen", "numbers", "value")


# ---------------------------------------------------------------------------
# I/O de imagem que funciona com acentos no caminho (ex.: "Área de Trabalho")
# ---------------------------------------------------------------------------
def ler_imagem(caminho: str | Path) -> np.ndarray | None:
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    if dados.size == 0:
        return None
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)


def salvar_imagem(caminho: str | Path, img: np.ndarray) -> None:
    ext = Path(caminho).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(caminho))


# ---------------------------------------------------------------------------
# Mapeamento das classes do modelo
# ---------------------------------------------------------------------------
def mapear_classes(nomes: dict[int, str]) -> tuple[dict[int, int], set[int]]:
    """Retorna ({id_classe: valor_digito}, {ids_de_visor})."""
    digitos, visor = {}, set()
    for cid, nome in nomes.items():
        n = str(nome).strip().lower()
        if re.fullmatch(r"\d", n):
            digitos[cid] = int(n)
        elif any(p in n for p in PALAVRAS_VISOR):
            visor.add(cid)
        else:
            # nomes tipo "digit_3", "num-7", "d5", "three"...
            m = re.search(r"(\d)$", n) or re.search(r"^(\d)\b", n)
            if m:
                digitos[cid] = int(m.group(1))
            else:
                extenso = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                           "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                           "um": 1, "dois": 2, "tres": 3, "três": 3, "quatro": 4,
                           "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9}
                if n in extenso:
                    digitos[cid] = extenso[n]
    return digitos, visor


# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------
@dataclass
class Digito:
    x1: float
    y1: float
    x2: float
    y2: float
    valor: int
    conf: float
    rolando: bool = False          # estava entre dois números
    alternativa: int | None = None

    @property
    def cx(self): return (self.x1 + self.x2) / 2

    @property
    def cy(self): return (self.y1 + self.y2) / 2

    @property
    def w(self): return self.x2 - self.x1

    @property
    def h(self): return self.y2 - self.y1


@dataclass
class Resultado:
    arquivo: str
    leitura: str = ""                # ex.: "00123,456"
    digitos_str: str = ""            # ex.: "00123456" ('?' = faltando)
    confianca_media: float = 0.0
    confianca_minima: float = 0.0
    completa: bool = False
    avisos: list[str] = field(default_factory=list)
    digitos: list[Digito] = field(default_factory=list)
    caixa_visor: tuple | None = None
    usou_recorte: bool = False

    @property
    def valor_m3(self) -> float | None:
        if "?" in self.leitura or not self.leitura:
            return None
        try:
            return float(self.leitura.replace(",", "."))
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Funções geométricas
# ---------------------------------------------------------------------------
def _iou(a: Digito, b: Digito) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    uni = a.w * a.h + b.w * b.h - inter
    return inter / uni if uni > 0 else 0.0


def _sobreposicao_horizontal(a: Digito, b: Digito) -> float:
    """Fração da largura da menor caixa que se sobrepõe horizontalmente."""
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    return ix / max(1e-6, min(a.w, b.w))


def remover_duplicados(cands: list[Digito], iou_lim: float, regra: str) -> list[Digito]:
    """NMS entre classes. Se 2 dígitos consecutivos disputam a mesma posição
    (dígito rolando), aplica a regra configurada."""
    cands = sorted(cands, key=lambda d: d.conf, reverse=True)
    finais: list[Digito] = []
    usados = [False] * len(cands)
    for i, d in enumerate(cands):
        if usados[i]:
            continue
        grupo = [d]
        for j in range(i + 1, len(cands)):
            if usados[j]:
                continue
            o = cands[j]
            if _iou(d, o) > iou_lim or (
                _sobreposicao_horizontal(d, o) > 0.6 and abs(d.cy - o.cy) < 0.5 * max(d.h, o.h)
            ):
                grupo.append(o)
                usados[j] = True
        escolhido = grupo[0]
        # segundo candidato com classe diferente e confiança relevante
        outros = [g for g in grupo[1:] if g.valor != escolhido.valor]
        if outros:
            seg = outros[0]
            consecutivos = (seg.valor - escolhido.valor) % 10 in (1, 9)
            if consecutivos and seg.conf >= 0.5 * escolhido.conf:
                a, b = escolhido.valor, seg.valor
                # menor na sequência circular: entre 9 e 0 -> 9
                menor = 9 if {a, b} == {9, 0} else min(a, b)
                if regra == "menor" and menor != escolhido.valor:
                    escolhido = Digito(escolhido.x1, escolhido.y1, escolhido.x2, escolhido.y2,
                                       menor, seg.conf, rolando=True, alternativa=escolhido.valor)
                else:
                    escolhido.rolando = True
                    escolhido.alternativa = seg.valor
        finais.append(escolhido)
    return finais


def agrupar_linha_principal(digs: list[Digito]) -> list[Digito]:
    """Mantém apenas os dígitos da linha do contador (maior grupo alinhado
    com alturas parecidas). Descarta nº de série, marca, etc."""
    if len(digs) <= 2:
        return digs
    melhor, melhor_score = [], -1.0
    # testa cada dígito como "semente" da linha
    for semente in digs:
        grupo = [d for d in digs
                 if 0.6 < d.h / semente.h < 1.6 and abs(d.cy - semente.cy) < 0.6 * semente.h]
        if len(grupo) >= 2:
            # refina com a reta ajustada (considera inclinação)
            pts = np.array([[d.cx, d.cy] for d in grupo], dtype=np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
            hmed = float(np.median([d.h for d in grupo]))
            grupo = [d for d in digs
                     if abs((d.cx - x0) * vy - (d.cy - y0) * vx) < 0.5 * hmed
                     and 0.6 < d.h / hmed < 1.6]
        score = sum(d.conf for d in grupo)
        if score > melhor_score:
            melhor, melhor_score = grupo, score
    return melhor


def ordenar(digs: list[Digito]) -> list[Digito]:
    """Ordena seguindo a direção da linha (robusto a rotação)."""
    if len(digs) < 2:
        return digs
    pts = np.array([[d.cx, d.cy] for d in digs], dtype=np.float32)
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    if vx < 0:
        vx, vy = -vx, -vy
    return sorted(digs, key=lambda d: d.cx * vx + d.cy * vy)


def montar_string(digs: list[Digito]) -> tuple[str, int]:
    """Monta a string e insere '?' onde o espaçamento indica dígito faltando."""
    if not digs:
        return "", 0
    if len(digs) < 3:
        return "".join(str(d.valor) for d in digs), 0
    centros = np.array([[d.cx, d.cy] for d in digs])
    dist = np.linalg.norm(np.diff(centros, axis=0), axis=1)
    passo = float(np.median(dist))
    s, faltando = str(digs[0].valor), 0
    for k, dd in enumerate(dist):
        n = int(round(dd / passo)) - 1 if passo > 0 else 0
        if n > 0 and dd > 1.6 * passo:
            s += "?" * n
            faltando += n
        s += str(digs[k + 1].valor)
    return s, faltando


def fracao_vermelha(img: np.ndarray, d: Digito) -> float:
    """Fração de pixels avermelhados no dígito (número vermelho ou fundo vermelho).
    Usa vermelho RELATIVO (R bem maior que G e B), que funciona com traço fino,
    luz amarelada e foto escura."""
    H, W = img.shape[:2]
    mx, my = d.w * 0.08, d.h * 0.08
    x1, y1 = int(max(0, d.x1 + mx)), int(max(0, d.y1 + my))
    x2, y2 = int(min(W, d.x2 - mx)), int(min(H, d.y2 - my))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    c = img[y1:y2, x1:x2].astype(np.int32)
    b, g, r = c[..., 0], c[..., 1], c[..., 2]
    return float(((r - np.maximum(g, b) > 35) & (r > 60)).mean())


# Padrão dos hidrômetros brasileiros: total de dígitos -> dígitos vermelhos
# (6 = 4 pretos + 2 vermelhos; 8 = 5 pretos + 3 vermelhos)
VERMELHOS_PADRAO = {6: 2, 8: 3}


def inteiros_pela_cor(img: np.ndarray, digs: list[Digito], s: str) -> tuple[int | None, str]:
    """Quantos caracteres de `s` são a parte inteira (m³): tudo antes dos dígitos
    vermelhos. Retorna (posição da vírgula ou None, origem: 'cor'|'padrao'|'')."""
    # posição de cada dígito detectado dentro de `s` (s pode conter '?')
    pos = [i for i, ch in enumerate(s) if ch != "?"]
    sc = [fracao_vermelha(img, d) for d in digs]
    n = len(s)

    k0 = VERMELHOS_PADRAO.get(n)
    if k0:
        pre = [x for p, x in zip(pos, sc) if p < n - k0]
        suf = [x for p, x in zip(pos, sc) if p >= n - k0]
        if pre and suf and max(suf) > max(pre) + 0.003 and np.mean(suf) > np.mean(pre):
            return n - k0, "cor"

    # busca geral: últimos k dígitos claramente mais vermelhos que os anteriores
    melhor: tuple[int, float] | None = None
    for k in range(1, min(4, len(sc) - 2) + 1):
        pre, suf = sc[:-k], sc[-k:]
        if min(suf) > max(0.01, 2.5 * max(pre)):
            folga = min(suf) - max(pre)
            if melhor is None or folga > melhor[1]:
                melhor = (pos[len(sc) - k], folga)
    if melhor:
        return melhor[0], "cor"
    if k0:
        return n - k0, "padrao"
    return None, ""


# ---------------------------------------------------------------------------
# Leitor
# ---------------------------------------------------------------------------
class LeitorHidrometro:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        self.cfg = cfg
        caminho = Path(cfg["modelo"])
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {caminho}\n"
                "Treine primeiro com:  python treinar.py")
        self.modelo = YOLO(str(caminho))
        self.map_dig, self.cls_visor = mapear_classes(self.modelo.names)
        if not self.map_dig:
            raise ValueError(f"Nenhuma classe de dígito reconhecida em {self.modelo.names}")

    # -- detecção bruta ------------------------------------------------------
    def _detectar(self, img: np.ndarray):
        r = cast("Results", next(iter(self.modelo.predict(
            img, imgsz=self.cfg["imgsz"], conf=0.05, iou=0.7, max_det=300, verbose=False))))
        digs, visores = [], []
        caixas = r.boxes
        if caixas is None:
            return digs, visores
        xyxy = np.asarray(cast("torch.Tensor", caixas.xyxy).cpu())
        conf = np.asarray(cast("torch.Tensor", caixas.conf).cpu())
        cls = np.asarray(cast("torch.Tensor", caixas.cls).cpu()).astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            if k in self.map_dig:
                digs.append(Digito(float(x1), float(y1), float(x2), float(y2),
                                   self.map_dig[k], float(c)))
            elif k in self.cls_visor:
                visores.append((float(x1), float(y1), float(x2), float(y2), float(c)))
        return digs, visores

    def _processar(self, cands: list[Digito]) -> list[Digito]:
        c = self.cfg
        # candidatos fracos ajudam a detectar dígito rolando, mas só
        # entram na leitura se passarem da confiança mínima
        cands = [d for d in cands if d.conf >= c["confianca_minima"] * 0.5]
        digs = remover_duplicados(cands, c["iou_duplicado"], c["regra_digito_rolando"])
        digs = [d for d in digs if d.conf >= c["confianca_minima"]]
        digs = agrupar_linha_principal(digs)
        return ordenar(digs)

    def _pontuar(self, digs: list[Digito], faltando: int) -> float:
        if not digs:
            return 0.0
        # soma das confianças: mais dígitos encontrados com boa confiança = melhor
        score = float(sum(d.conf for d in digs)) - 0.5 * faltando
        n_esp = self.cfg.get("num_digitos")
        if n_esp:
            score -= abs(n_esp - (len(digs) + faltando))
        return score

    # -- leitura completa ----------------------------------------------------
    def ler(self, img: np.ndarray, nome: str = "") -> Resultado:
        c = self.cfg
        res = Resultado(arquivo=nome)
        cands, visores = self._detectar(img)

        opcoes = []  # (score, digs, faltando, caixa_visor, usou_recorte)
        digs = self._processar(cands)
        _, falt = montar_string(digs)
        opcoes.append((self._pontuar(digs, falt), digs, falt, None, False))

        visores = [v for v in visores if v[4] >= c["confianca_minima"]]
        if c.get("usar_recorte_visor", True) and visores:
            x1, y1, x2, y2, _ = max(visores, key=lambda v: v[4])
            H, W = img.shape[:2]
            m = c.get("margem_recorte", 0.15)
            mx, my = (x2 - x1) * m, (y2 - y1) * m
            rx1, ry1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
            rx2, ry2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
            recorte = img[ry1:ry2, rx1:rx2]
            if recorte.size and recorte.shape[0] > 8 and recorte.shape[1] > 8:
                cands_r, _ = self._detectar(recorte)
                for d in cands_r:  # volta p/ coordenadas da imagem original
                    d.x1 += rx1; d.x2 += rx1; d.y1 += ry1; d.y2 += ry1
                digs_r = self._processar(cands_r)
                _, falt_r = montar_string(digs_r)
                # leve preferência pelo recorte (dígitos maiores = mais confiável)
                opcoes.append((self._pontuar(digs_r, falt_r) * 1.05, digs_r, falt_r,
                               (rx1, ry1, rx2, ry2), True))

        _, digs, falt, caixa, recortou = max(opcoes, key=lambda o: o[0])
        res.digitos, res.caixa_visor, res.usou_recorte = digs, caixa, recortou

        if not digs:
            res.avisos.append("nenhum dígito detectado")
            return res

        s, falt = montar_string(digs)
        res.digitos_str = s
        res.confianca_media = float(np.mean([d.conf for d in digs]))
        res.confianca_minima = float(min(d.conf for d in digs))

        n_esp = c.get("num_digitos")
        if falt:
            res.avisos.append(f"{falt} dígito(s) possivelmente faltando")
        if n_esp and len(s) != n_esp:
            res.avisos.append(f"esperado {n_esp} dígitos, lido {len(s)}")
        rol = [i for i, d in enumerate(digs) if d.rolando]
        if rol:
            res.avisos.append(f"dígito rolando na posição {', '.join(str(i + 1) for i in rol)}")
        if res.confianca_minima < 0.5:
            res.avisos.append("baixa confiança em algum dígito - conferir")

        res.completa = "?" not in s and (not n_esp or len(s) == n_esp)

        ni = c.get("digitos_inteiros")
        if ni == "auto":
            ni, origem = inteiros_pela_cor(img, digs, s)
            if ni is None:
                res.avisos.append("não identifiquei os dígitos vermelhos - sem vírgula")
            elif origem == "padrao":
                res.avisos.append("vírgula pela quantidade de dígitos (cor inconclusiva)")
        if isinstance(ni, int) and 0 < ni < len(s) and (not n_esp or len(s) == n_esp):
            res.leitura = f"{s[:ni]},{s[ni:]}"
        else:
            res.leitura = s
        return res

    # -- desenho -------------------------------------------------------------
    @staticmethod
    def desenhar(img: np.ndarray, res: Resultado) -> np.ndarray:
        out = img.copy()
        esp = max(1, int(round(max(out.shape[:2]) / 600)))
        if res.caixa_visor:
            x1, y1, x2, y2 = res.caixa_visor
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 160, 0), esp)
        for d in res.digitos:
            cor = (0, 200, 0) if d.conf >= 0.5 else (0, 165, 255)
            if d.rolando:
                cor = (255, 0, 255)
            p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
            cv2.rectangle(out, p1, p2, cor, esp)
            cv2.putText(out, str(d.valor), (p1[0], max(0, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * esp + 0.3, cor, esp, cv2.LINE_AA)
        texto = f"Leitura: {res.leitura or '---'}  ({res.confianca_media:.0%})"
        escala = 0.6 * esp + 0.2
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, esp + 1)
        cv2.rectangle(out, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
        cv2.putText(out, texto, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, escala,
                    (255, 255, 255), esp + 1, cv2.LINE_AA)
        return out


def listar_imagens(pasta: str | Path) -> list[Path]:
    pasta = Path(pasta)
    if not pasta.exists():
        return []
    return sorted(p for p in pasta.rglob("*") if p.suffix.lower() in EXTENSOES_IMAGEM)


def carregar_config(caminho: str | Path = "config.yaml") -> dict:
    import yaml
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)
