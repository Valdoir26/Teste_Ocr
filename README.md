# Leitura de Hidrômetros com YOLO26

Lê automaticamente o consumo (m³) em fotos de hidrômetros. Cada dígito (0–9)
é detectado pelo YOLO26, que é o modelo mais recente da Ultralytics e o melhor
deles para objetos pequenos. Depois disso, o pós-processamento monta a leitura.

## Estrutura

```
Teste_Ocr/
├── imagens_treino/       <- fotos SEM rótulo para anotar (as suas 74 já estão aqui)
├── dataset/              <- datasets rotulados (o rotular.py grava em dataset/proprio/;
│                            datasets prontos de outras fontes também vão aqui)
├── imagens_entrada/      <- COLOQUE AQUI as fotos que devem ser lidas
├── resultados/           <- leituras.csv + imagens com os dígitos marcados
├── modelos/              <- modelo treinado (hidrometro_best.pt)
├── config.yaml           <- todas as configurações
├── rotular.py            <- 0. anota as fotos (4 cliques + digitar a leitura)
├── preparar_dataset.py   <- 1. organiza/une os datasets + gera sintéticas
├── treinar.py            <- 2. treina o YOLO
├── avaliar.py            <- 3. mede a precisão
├── ler_hidrometro.py     <- 4. lê as imagens
└── leitor.py             <- lógica de leitura (usado pelos scripts)
```

## Passo a passo

```powershell
pip install -r requirements.txt

# 0a. (se o dataset em dataset/ tiver só a classe "hidrometro", sem os dígitos)
#     aproveita as caixas já desenhadas no Roboflow
python importar_roboflow.py

# 0b. Anote as fotos de imagens_treino/ (aprox. 15 s por foto; 5 s se já importada)
python rotular.py

# 1. Monte o dataset (divide treino/validação e cria imagens sintéticas)
python preparar_dataset.py

# 2. Treine (RTX 3050 6GB: aprox. 1-3 h dependendo do nº de imagens)
python treinar.py

# 3. Veja a precisão real da leitura
python avaliar.py

# 4. Coloque as fotos em imagens_entrada/ e leia
python ler_hidrometro.py
```

## Como anotar (`rotular.py`)

1. **Arraste** um retângulo sobre o visor para dar zoom (a rodinha do mouse também dá zoom).
2. **Clique nos 4 cantos** da janela de dígitos, em qualquer ordem. Contorne só os
   dígitos, sem o "m³". Funciona com foto torta ou em perspectiva.
3. **Digite a leitura** completa, com todos os dígitos pretos e vermelhos, e tecle **Enter**.
   A ferramenta mostra como vai dividir as caixas antes de salvar.

- Dígito **rolando** (entre dois números): digite o **menor** (entre 3 e 4 → 3; entre 9 e 0 → 9).
- Foto ilegível, desenho ou propaganda sem hidrômetro legível: **Descartar**.
- Dá para fechar e continuar depois, e também voltar para corrigir (PgUp).
- Fotos **importadas** do Roboflow já abrem com as caixas ou os cantos marcados: só
  digite a leitura. Se as caixas estiverem incompletas ou a divisão sair torta
  (visor inclinado), tecle **ESC** e clique os 4 cantos reais.

> **Anotando no Roboflow?** Use uma classe por dígito (`0`, `1`, … `9`) e, se quiser,
> `visor` para a janela inteira. Assim o dataset entra direto, sem passar pelo `rotular.py`.
> Uma classe única como `hidrometro` não ensina o modelo a *ler*, só a *achar* os dígitos.

`preparar_dataset.py` cria por padrão 10 imagens sintéticas por foto de treino, trocando
os dígitos de lugar dentro do visor e sorteando mais os dígitos raros. Para mudar a
quantidade: `python preparar_dataset.py --sinteticas 20` (ou `0` para desligar).
Fotos do mesmo hidrômetro ficam sempre juntas no treino **ou** na validação, para
a precisão medida não sair inflada.

## Formato do dataset

Aceita **um ou vários** datasets dentro de `dataset/`, em qualquer um destes formatos:

- exportação do Roboflow no formato **YOLOv8 / YOLO11** (pastas `train/ valid/ test/` + `data.yaml`)
- pastas `images/` + `labels/`
- imagens e `.txt` na mesma pasta, com `classes.txt`

Nomes de classe aceitos:
- **dígitos:** `0`…`9`, `digit_3`, `num-7`, `three`, e variações parecidas
- **visor (opcional):** `display`, `counter`, `visor`, `contador`, `reading`, etc.

Classes com outros nomes são ignoradas. Rótulos em polígono (segmentação/OBB)
são convertidos em caixas.

> Se o dataset tiver a classe do **visor**, a leitura passa a ser feita em 2 etapas:
> recorta o visor e detecta os dígitos no recorte ampliado. Com isso a precisão sobe.

## Configurações importantes (`config.yaml`)

| Opção | O que faz |
|---|---|
| `num_digitos` | total de dígitos do mostrador (ex.: `8`). **Configure se todos os seus hidrômetros tiverem o mesmo nº de dígitos**, porque isso aumenta muito a confiabilidade |
| `digitos_inteiros` | `auto` (padrão) = põe a vírgula antes dos dígitos **vermelhos**, detectados pela cor, e serve para modelos com 4, 5 ou 6 dígitos pretos. Ou um número fixo, ex.: `5` → `00123,456` |
| `regra_digito_rolando` | `menor` = quando o dígito está entre dois números, usa o menor (padrão das concessionárias) |
| `confianca_minima` | abaixe (0.15) se estiver faltando dígito; suba (0.4) se aparecer dígito a mais |
| `modelo_base` (treino) | `yolo26s.pt` (padrão). Use `yolo26m.pt` para mais precisão se tiver muitas imagens |

## Resultado (`resultados/leituras.csv`)

Abre direto no Excel. Colunas: `arquivo`, `leitura`, `valor_m3`, `confianca_media`,
`completa`, **`conferir`**, `avisos`.

A coluna **`conferir = sim`** marca as leituras suspeitas, que são as de baixa confiança,
com dígito faltando (`?`) ou com dígito rolando. Revise essas manualmente.

Nas imagens anotadas: **verde** = confiante, **laranja** = baixa confiança,
**roxo** = dígito rolando, **azul** = visor.

## Dicas para máxima precisão

1. **Dados são o mais importante:** use pelo menos 150–300 exemplos de cada dígito,
   com fotos variadas (iluminação, ângulo, sujeira, reflexo). Rode `preparar_dataset.py`
   e veja o resumo por classe.
2. Rode `avaliar.py` e abra `resultados/erros_avaliacao/`. Os erros mostram que tipo de
   foto falta no dataset.
3. Se os dígitos forem pequenos na foto, treine e leia com `imgsz: 960`
   (se faltar memória, use `batch: 8`).
4. Se o dataset for grande (>2000 imagens), teste `python treinar.py --modelo yolo26m.pt`.
5. Configure `num_digitos` no `config.yaml`.
6. **OneDrive:** esta pasta está no OneDrive, e a sincronização deixa o treino mais lento
   (o YOLO avisa "Slow image access"). Pause a sincronização durante o treino ou copie
   o projeto para uma pasta local (ex.: `C:\hidrometro`).
7. Um treino interrompido pode ser retomado com `python treinar.py --continuar`.
