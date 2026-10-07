"""Gera o relatorio CP5 de tres paginas a partir dos resultados versionados."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
BLUE = colors.HexColor("#10344E")
LIGHT = colors.HexColor("#EDF5F8")
MUTED = colors.HexColor("#445C6B")
regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
if not regular.exists():
    regular = Path(reportlab.__file__).parent / "fonts/Vera.ttf"
    bold = Path(reportlab.__file__).parent / "fonts/VeraBd.ttf"
pdfmetrics.registerFont(TTFont("DV", str(regular)))
pdfmetrics.registerFont(TTFont("DVB", str(bold)))
BODY = ParagraphStyle("body", fontName="DV", fontSize=8.4, leading=11.5, textColor=BLUE, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.1, leading=9.5)
TITLE = ParagraphStyle("title", parent=BODY, fontName="DVB", fontSize=21, leading=25, spaceAfter=12)
HEADING = ParagraphStyle("heading", parent=BODY, fontName="DVB", fontSize=11.3, leading=14, spaceBefore=7, spaceAfter=5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=6.9, leading=9, spaceAfter=0)
CAPTION = ParagraphStyle("caption", parent=SMALL, fontName="DVB", alignment=TA_CENTER)


def p(text, style=BODY):
    return Paragraph(text, style)


def table(rows, widths, small=False):
    style = CELL if small else SMALL
    cells = [[p(escape(str(value)), style) for value in row] for row in rows]
    result = Table(cells, colWidths=[width * cm for width in widths], repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AFC6D2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return result


def footer(canvas, doc):
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(BLUE)
    canvas.rect(0, height - 1.10 * cm, width, 1.10 * cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("DVB", 8)
    canvas.drawString(1.4 * cm, height - 0.68 * cm, "APPLIED COMPUTER VISION 2026 | CHECKPOINT 5")
    canvas.setFillColor(MUTED)
    canvas.setFont("DV", 7)
    canvas.drawString(1.4 * cm, 0.65 * cm, "YOLO + U-Net + SlowFast | Inferencia sem treinamento adicional")
    canvas.drawRightString(width - 1.4 * cm, 0.65 * cm, f"{doc.page}/3")
    canvas.restoreState()


def load(stem):
    base = ROOT / "docs/results"
    with (base / f"{stem}_frames.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    predictions = json.loads((base / f"{stem}_janelas.json").read_text())
    comparisons = json.loads((base / "isolados" / f"{stem}_comparacao.json").read_text())
    with (base / "isolados" / f"{stem}_isolados.csv").open(encoding="utf-8") as stream:
        isolated = list(csv.DictReader(stream))
    return rows, predictions, comparisons, isolated


def image_strip(stem, height):
    images = [Image(str(ROOT / f"docs/images/evidencias/{stem}_anotado_momento_{i}.jpg"),
                    width=5.6 * cm, height=height * cm) for i in (1, 2, 3)]
    return Table([images], colWidths=[5.8 * cm] * 3)


def build():
    a, pa, ca, _ia = load("video_1_archery")
    b, pb, cb, ib = load("video_2_theatre")
    manual = json.loads((ROOT / "docs/results/avaliacao_manual.json").read_text())
    na = len(a)
    nb = len(b)
    ma = sum(int(row["area_mascara_px"]) > 0 for row in a)
    mb = sum(int(row["area_mascara_px"]) > 0 for row in b)
    da = sum(row["pessoa_detectada"] == "True" for row in a)
    db = sum(row["pessoa_detectada"] == "True" for row in b)
    mib = sum(int(row["mascara_unet_px"]) > 0 for row in ib)
    story = [
        p("Detecção de ações de pessoas", TITLE),
        p("Localização, contorno e ação em dois vídeos curtos. Relatório revisado com resultados reais do pipeline corrigido."),
        p("Objetivo e modelos", HEADING),
        p("Integrar modelos já treinados, sem treinar ou ajustar pesos: <b>YOLO11n</b> (COCO, classe person), "
          "<b>U-Net com encoder EfficientNet-B0</b> e decoder treinados para segmentação humana, e "
          "<b>SlowFast R50</b> (Kinetics-400). A U-Net completa substitui a U²-Net utilizada na versão anterior."),
        p("Integração e cobertura temporal", HEADING),
        p("O YOLO fornece as caixas. Cada região é expandida em 10%, recortada e enviada à U-Net em RGB, "
          "320 × 320 e valores [0,1]. O sigmoid e limiar 0,5 produzem a máscara, redimensionada e reprojetada "
          "no quadro original. A seleção principal prioriza candidatos com máscara não vazia e continuidade "
          "por IoU, com confiança e área como alternativa. Isso reduz a seleção do fundo, mas não garante identidade."),
        p("O SlowFast recebe 32 quadros, passo 16 e α=4. A última janela é fechada no último quadro, mesmo "
          "fora do passo regular. Em uma segunda passagem offline, cada quadro recebe a previsão da janela "
          "que o contém e tem o centro mais próximo. Assim a ação aparece desde o início, usando contexto "
          "temporal futuro; não se trata de processamento em tempo real. As saídas são MP4, CSV e JSON com top-3."),
        p("Execução isolada e experimentos", HEADING),
        p(f"Cada modelo também foi executado separadamente nos dois trechos. A U-Net isolada recebe o quadro "
          f"inteiro, sem YOLO; a execução integrada recebe os recortes. Foram analisados {na} e {nb} quadros "
          "a 8 FPS, por 10 s. O SlowFast possui quatro janelas em cada vídeo: 0-4, 2-6, 4-8 e 6-10 s. "
          "O teste opcional usa as mesmas caixas do pipeline para comparar contexto completo e recorte."),
        p("Grupo e responsabilidades", HEADING),
        table([
            ["Integrante", "RM", "Área / parecer técnico"],
            ["Guilherme Santos", "551168", "Integração: cobertura completa e dados reproduzíveis são essenciais."],
            ["Enricco", "551717", "YOLO: várias pessoas e entrada no quadro exigem avaliar o alvo escolhido."],
            ["Gabriel", "99227", "U-Net: máscara não vazia não significa contorno correto; o recorte altera o resultado."],
            ["Danilo", "99465", "SlowFast: confiança é saída do modelo e não mede acurácia ou certeza."],
            ["Laura", "98747", "Análise: manter os erros permite explicar limites e comparar as janelas."],
        ], [3.4, 1.6, 12.4]),
        p("Fontes e reprodução", HEADING),
        p("Vídeo 1: archery.mp4 do tutorial público PyTorchVideo (0-10 s). Vídeo 2: theatre.webm, "
          "Theatre Flamenco of San Francisco, do tutorial de detecção PyTorchVideo (1-11 s). Trechos, URLs "
          "diretas e checksums estão versionados. O checkpoint U-Net é de Aman Gupta, repositório "
          "PyTorch-Image-Segmentation, revisão d5f9ab4, licença MIT. É independente do arquivo do Teams.", SMALL),
        p("Reprodução: instalar requirements-cpu.txt e executar <b>python scripts/reproduce_cp5.py</b>. "
          "O repositório inclui dados brutos, execuções isoladas, configuração e manifesto de ambiente.", SMALL),
        PageBreak(),
        p("Resultados e inspeção visual", TITLE),
        p(f"YOLO: {da}/{na} quadros com pessoa no arco e flecha; {db}/{nb} no palco. Máscara integrada "
          f"não vazia: {ma}/{na} e {mb}/{nb}. Contagens indicam presença de saída, não acurácia ou qualidade de contorno."),
    ]
    rows = [["V.", "Tempo", "Caixa", "Máscara (px)", "Ação prevista", "Conf.", "Inspeção visual"]]
    for number, stem, data in [(1, "video_1_archery", a), (2, "video_2_theatre", b)]:
        for index in (0, 40, 79):
            row = data[index]
            observation = manual[stem][str(index)]
            rows.append([
                number, f"{float(row['tempo_s']):.2f}s", observation["caixa"], row["area_mascara_px"],
                row["acao"], f"{float(row['confianca_slowfast'])*100:.1f}%", observation["parecer"],
            ])
    story += [
        table(rows, [1.0, 1.3, 1.4, 1.8, 3.9, 1.6, 6.4], small=True),
        Spacer(1, 0.22 * cm),
        p("Vídeo 1: preparação (0 s), mira (5 s) e recuperação após disparo (9,88 s)", CAPTION),
        image_strip("video_1_archery", 4.2),
        Spacer(1, 0.12 * cm),
        p("Vídeo 2: entrada parcial/borrada, dança no meio do trecho e seleção do fundo", CAPTION),
        image_strip("video_2_theatre", 3.15),
        p("A caixa acompanha o arqueiro selecionado, embora extremidades fiquem fora do quadro. A máscara "
          "cobre o corpo apenas parcialmente e inclui objetos ou fundo. No palco, a seleção melhora em "
          "parte do trecho, mas volta a uma pessoa ao fundo quando nenhum candidato é segmentável.", SMALL),
        p("A ação observada no vídeo 2 é dança de flamenco. Tango é uma aproximação do gênero; "
          "giving or receiving award não corresponde à ação observada. Flamenco não aparece como classe "
          "específica no vocabulário Kinetics-400 utilizado. Não foram calculados IoU/Dice ou acurácia: "
          "não há máscaras de referência nem anotações completas dos vídeos.", SMALL),
        PageBreak(),
        p("Erros, transições e comparação", TITLE),
        p("Erro 1 - associação do alvo e falha de segmentação", HEADING),
        p(f"No palco, a seleção pode persistir em um participante ao fundo, e a máscara integrada é não vazia "
          f"em {mb}/{nb} quadros. A U-Net isolada produz máscara em {mib}/{nb}, o que impede atribuir a falha "
          "apenas à iluminação. A seleção do alvo, a escala e a deformação do recorte também interferem. "
          "Priorizar máscaras válidas melhora parte do trecho, mas não substitui rastreamento com identidade."),
        p("Erro 2 - classe incompatível no palco", HEADING),
        p("Na janela 4-8 s, o SlowFast prevê giving or receiving award, apesar de a pessoa continuar dançando. "
          "Outras janelas retornam tango dancing, que não identifica corretamente o flamenco. Contexto de palco, "
          "mudança de pose e vocabulário limitado são hipóteses explicativas; não foi feita intervenção controlada "
          "para provar a causa. Aumentar a confiança não corrige automaticamente a classe."),
        p("Início, execução e término: a confiança cai ou a classe muda?", HEADING),
        p("No arco e flecha, preparação, mira e recuperação após disparo aparecem no trecho. A classe permanece "
          "archery e a confiança fica em 100% nas quatro janelas, inclusive na janela final: não há queda observada "
          "ou troca de classe nas fases visuais. Isso mostra que uma janela pode conservar contexto da ação anterior. "
          "No palco, há movimento contínuo; o trecho não documenta um início ou término completo do flamenco. "
          "A classe muda mesmo assim, portanto essa mudança não pode ser apresentada como prova de início ou fim."),
    ]
    windows = [["Janela (s)", "Arco e flecha", "Palco: classe / confiança"]]
    for first, second in zip(pa, pb):
        windows.append([
            f"{first['inicio_s']:.0f}-{first['fim_s']:.0f}",
            f"{first['acao']} / {first['confianca']*100:.1f}%",
            f"{second['acao']} / {second['confianca']*100:.1f}%",
        ])
    story += [table(windows, [2.2, 5.0, 10.2], small=True), p("Extensão opcional - quadro e recorte", HEADING)]
    comparison = [["Vídeo / janela", "Quadro inteiro", "Recorte da pessoa"]]
    for number, values in [(1, ca), (2, cb)]:
        for item in values:
            full, crop = item["quadro_inteiro"][0], item["recorte_pessoa"][0]
            comparison.append([
                f"V{number} / {item['inicio_s']:.0f}-{item['fim_s']:.0f}s",
                f"{full['acao']} / {full['confianca']*100:.1f}%",
                f"{crop['acao']} / {crop['confianca']*100:.1f}%",
            ])
    story += [
        table(comparison, [2.2, 7.6, 7.6], small=True),
        p("O recorte usa a união das caixas da mesma janela. Pode reduzir contexto ou incluir o alvo incorreto; "
          "a comparação não prova que recortar sempre melhora o reconhecimento.", SMALL),
        p("Conclusão", HEADING),
        p("Os três modelos pré-treinados foram integrados e executados isoladamente, sem treinamento adicional. "
          "A demonstração conserva caixa, máscara, ação e confiança, inclusive resultados ruins. O relatório responde "
          "às diferenças de localização, contorno e classe nos seis momentos; CSV, JSON, vídeos e scripts permitem "
          "inspecionar as conclusões. Os pareceres técnicos do grupo devem ser revisados pelos integrantes antes da entrega.", SMALL),
    ]
    document = SimpleDocTemplate(
        str(ROOT / "docs/relatorio_cp5.pdf"), pagesize=A4,
        leftMargin=1.4*cm, rightMargin=1.4*cm, topMargin=1.5*cm, bottomMargin=1.1*cm,
        title="CP5 - YOLO, U-Net e SlowFast - resultados revisados",
        author="Guilherme Santos, Enricco, Gabriel, Danilo e Laura",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
