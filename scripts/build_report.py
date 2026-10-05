"""Gera o relatorio final do CP5 em exatamente tres paginas."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "relatorio_cp5.pdf"
BLUE = colors.HexColor("#0B2A4A")
CYAN = colors.HexColor("#18A4C7")
LIGHT = colors.HexColor("#EAF4F8")
MUTED = colors.HexColor("#526477")
GREEN = colors.HexColor("#1F8A70")
ORANGE = colors.HexColor("#E67E22")


def register_fonts() -> tuple[str, str]:
    regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    pdfmetrics.registerFont(TTFont("DV", regular))
    pdfmetrics.registerFont(TTFont("DV-Bold", bold))
    return "DV", "DV-Bold"


FONT, FONT_BOLD = register_fonts()


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def load_results(stem: str) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    with (ROOT / "outputs" / f"{stem}_frames.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    predictions = json.loads((ROOT / "outputs" / f"{stem}_janelas.json").read_text(encoding="utf-8"))
    return rows, predictions


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(BLUE)
    canvas.rect(0, height - 1.25 * cm, width, 1.25 * cm, fill=1, stroke=0)
    canvas.setFont(FONT_BOLD, 8.5)
    canvas.setFillColor(colors.white)
    canvas.drawString(1.45 * cm, height - 0.78 * cm, "APPLIED COMPUTER VISION 2026 | CHECKPOINT 5")
    canvas.setFillColor(MUTED)
    canvas.setFont(FONT, 8)
    canvas.drawString(1.45 * cm, 0.72 * cm, "YOLO + U-Net + SlowFast | Modelos pre-treinados, sem treinamento adicional")
    canvas.drawRightString(width - 1.45 * cm, 0.72 * cm, f"{doc.page} / 3")
    canvas.restoreState()


def flow_table(body_style: ParagraphStyle) -> Table:
    labels = [
        ("1", "YOLO", "pessoa e caixa"),
        ("2", "U-Net", "mascara no recorte"),
        ("3", "SlowFast", "acao por janela"),
        ("4", "Saida", "MP4 + CSV + JSON"),
    ]
    cells = []
    for number, title, subtitle in labels:
        cells.append(
            paragraph(
                f"<font color='#18A4C7'><b>{number}</b></font><br/><b>{title}</b><br/>"
                f"<font color='#526477' size='8'>{subtitle}</font>",
                body_style,
            )
        )
    table = Table([cells], colWidths=[4.35 * cm] * 4, rowHeights=[2.05 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.8, CYAN),
                ("INNERGRID", (0, 0), (-1, -1), 0.8, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def evidence_strip(stem: str, label: str, caption_style: ParagraphStyle) -> KeepTogether:
    images = []
    for index in range(1, 4):
        path = ROOT / "docs" / "images" / "evidencias" / f"{stem}_anotado_momento_{index}.jpg"
        images.append(Image(str(path), width=5.65 * cm, height=4.0 * cm))
    table = Table([images], colWidths=[5.8 * cm] * 3)
    table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return KeepTogether([paragraph(label, caption_style), Spacer(1, 0.08 * cm), table])


def build() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleDV", parent=styles["Title"], fontName=FONT_BOLD, fontSize=24, leading=29, textColor=BLUE, alignment=TA_LEFT
    )
    subtitle = ParagraphStyle(
        "SubtitleDV", parent=styles["Normal"], fontName=FONT, fontSize=11, leading=16, textColor=MUTED
    )
    heading = ParagraphStyle(
        "HeadingDV", parent=styles["Heading2"], fontName=FONT_BOLD, fontSize=15, leading=19, textColor=BLUE, spaceBefore=5, spaceAfter=7
    )
    body = ParagraphStyle(
        "BodyDV", parent=styles["BodyText"], fontName=FONT, fontSize=9.2, leading=13.4, textColor=colors.HexColor("#1D2A36")
    )
    small = ParagraphStyle(
        "SmallDV", parent=body, fontSize=7.6, leading=10.5, textColor=MUTED
    )
    caption = ParagraphStyle(
        "CaptionDV", parent=body, fontName=FONT_BOLD, fontSize=8.2, leading=10, textColor=BLUE, alignment=TA_CENTER
    )
    card = ParagraphStyle(
        "CardDV", parent=body, fontSize=8.5, leading=12, alignment=TA_CENTER
    )

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=1.45 * cm,
        rightMargin=1.45 * cm,
        topMargin=1.75 * cm,
        bottomMargin=1.25 * cm,
        title="CP5 - Deteccao de Acoes com YOLO, U-Net e SlowFast",
        author="Guilherme Santos, Enricco, Gabriel, Danilo e Laura",
    )
    story = []

    # Pagina 1 - objetivo e metodo
    story.extend(
        [
            Spacer(1, 0.22 * cm),
            paragraph("Detecção de Ações de Pessoas", title),
            paragraph("Integração de YOLO, U-Net e SlowFast em vídeos curtos", subtitle),
            Spacer(1, 0.32 * cm),
            Table(
                [[
                    paragraph("<font color='#FFFFFF'>CHECKPOINT 5</font>", caption),
                    paragraph("<font color='#FFFFFF'>ENTREGA: 06/10/2026</font>", caption),
                    paragraph("<font color='#FFFFFF'>CÓDIGO EXECUTÁVEL</font>", caption),
                ]],
                colWidths=[5.8 * cm] * 3,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), BLUE),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                        ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
                        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            ),
            Spacer(1, 0.3 * cm),
            paragraph("Objetivo", heading),
            paragraph(
                "Desenvolver um programa que localize a pessoa, segmente seus pixels e classifique a ação em janelas temporais sucessivas. "
                "Foram integrados exclusivamente modelos já treinados: YOLO11n (COCO), U2-Net Human Segmentation e SlowFast R50 (Kinetics-400).",
                body,
            ),
            Spacer(1, 0.25 * cm),
            flow_table(card),
            Spacer(1, 0.25 * cm),
            paragraph("Implementação", heading),
            paragraph(
                "O YOLO filtra a classe <i>person</i>. A detecção principal combina confiança e área; sua caixa é expandida em 10% e enviada à U-Net. "
                "A máscara binária é redimensionada e reprojetada nas coordenadas do quadro original. O SlowFast recebe janelas de 32 quadros, "
                "com passo 16 e razão temporal α=4. Cada quadro final reúne caixa, máscara, ação e confiança. Também são gerados CSV por quadro e JSON por janela.",
                body,
            ),
            Spacer(1, 0.25 * cm),
            paragraph("Grupo e responsabilidades", heading),
        ]
    )
    team = [
        ["Integrante", "RM", "Responsabilidade"],
        ["Guilherme Santos", "551168", "Integração, testes e documentação"],
        ["Enricco", "551717", "YOLO e leitura dos vídeos"],
        ["Gabriel", "99227", "U-Net e projeção da máscara"],
        ["Danilo", "99465", "SlowFast e janelas temporais"],
        ["Laura", "98747", "Experimentos, análise e apresentação"],
    ]
    team_table = Table(team, colWidths=[4.2 * cm, 2.1 * cm, 11.1 * cm], repeatRows=1)
    team_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("FONTNAME", (0, 1), (-1, -1), FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B9CBD8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend([team_table, PageBreak()])

    # Pagina 2 - resultados medidos
    archery_rows, archery_pred = load_results("video_1_archery")
    theatre_rows, theatre_pred = load_results("video_2_theatre")
    story.extend(
        [
            paragraph("Experimentos e resultados", title),
            paragraph(
                "Foram processados dois clipes públicos de 8 s a 8 FPS (64 quadros): arco e flecha em ambiente externo e dança em palco escuro. "
                "As amostras abaixo correspondem ao fim das três janelas temporais.",
                body,
            ),
            Spacer(1, 0.25 * cm),
        ]
    )
    cards = Table(
        [[
            paragraph("<b>VÍDEO 1</b><br/><font size='15' color='#1F8A70'>64/64</font><br/>detecções YOLO", card),
            paragraph("<b>YOLO MÉDIO</b><br/><font size='15' color='#1F8A70'>83,5%</font><br/>arco e flecha", card),
            paragraph("<b>VÍDEO 2</b><br/><font size='15' color='#E67E22'>12/64</font><br/>máscaras não vazias", card),
            paragraph("<b>SLOWFAST</b><br/><font size='15' color='#E67E22'>88,8 → 75,0%</font><br/>dança no palco", card),
        ]],
        colWidths=[4.35 * cm] * 4,
        rowHeights=[1.75 * cm],
    )
    cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.7, CYAN),
                ("INNERGRID", (0, 0), (-1, -1), 0.7, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.extend([cards, Spacer(1, 0.3 * cm)])

    result_data = [["Vídeo", "Momento", "YOLO", "Máscara (px)", "Ação SlowFast", "Conf.", "Avaliação"]]
    indices = [32, 47, 63]
    for video_label, rows, predictions, expected in [
        ("1", archery_rows, archery_pred, "archery"),
        ("2", theatre_rows, theatre_pred, "dancing ballet"),
    ]:
        for position, (frame_index, prediction) in enumerate(zip(indices, predictions), start=1):
            row = rows[frame_index]
            mask_area = int(row["area_mascara_px"])
            mask_state = str(mask_area) if mask_area else "0 (falha)"
            evaluation = "correta" if prediction["acao"] == expected else "divergente"
            result_data.append(
                [
                    video_label,
                    f"{position} ({float(row['tempo_s']):.1f}s)",
                    f"{float(row['confianca_yolo']) * 100:.1f}%",
                    mask_state,
                    str(prediction["acao"]),
                    f"{float(prediction['confianca']) * 100:.1f}%",
                    evaluation,
                ]
            )
    results_table = Table(result_data, colWidths=[0.8 * cm, 2.1 * cm, 1.55 * cm, 2.35 * cm, 3.55 * cm, 1.55 * cm, 2.2 * cm], repeatRows=1)
    results_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("FONTNAME", (0, 1), (-1, -1), FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 7.1),
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9CBD8")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TEXTCOLOR", (3, 4), (3, 6), colors.HexColor("#B8422D")),
            ]
        )
    )
    story.extend(
        [
            results_table,
            Spacer(1, 0.2 * cm),
            evidence_strip("video_1_archery", "Vídeo 1 - três momentos de arco e flecha", caption),
            Spacer(1, 0.14 * cm),
            evidence_strip("video_2_theatre", "Vídeo 2 - três momentos de dança em palco", caption),
            Spacer(1, 0.08 * cm),
            paragraph(
                "Fonte dos clipes: exemplos públicos do projeto PyTorchVideo. As imagens mostram as saídas efetivamente produzidas nesta execução.",
                small,
            ),
            PageBreak(),
        ]
    )

    # Pagina 3 - falhas, parecer e conclusao
    story.extend(
        [
            paragraph("Análise de falhas e conclusão", title),
            paragraph("Erro 1 - seleção da pessoa em cenas com múltiplos indivíduos", heading),
            paragraph(
                "Nos dois vídeos há mais de uma pessoa. A heurística escolhe uma única detecção usando confiança × √área; por isso, pode trocar o alvo ou "
                "preferir uma pessoa secundária. No vídeo 1, a ação continuou correta porque o contexto de arco e flecha é forte, mas a caixa nem sempre "
                "representou a pessoa mais próxima. Uma correção futura é incorporar rastreamento (ByteTrack) e manter o mesmo ID entre quadros.",
                body,
            ),
            paragraph("Erro 2 - máscara ausente em iluminação baixa", heading),
            paragraph(
                "No palco escuro, o YOLO detectou pessoa em 64/64 quadros (80,3% de confiança média), porém a U-Net produziu máscara não vazia em apenas "
                "12/64 quadros. A baixa separação entre roupa, fundo e sombra, somada à pessoa pequena, prejudicou o contorno. Ajustar o limiar reduz falsos "
                "negativos, mas pode ampliar ruído; um checkpoint treinado no domínio do palco seria mais adequado.",
                body,
            ),
            paragraph("Transições temporais", heading),
            paragraph(
                "A classe <i>archery</i> permaneceu em 100% nas três janelas. No segundo vídeo, <i>dancing ballet</i> permaneceu correta, mas caiu de 88,8% "
                "para 83,7% e 75,0%. A queda acompanha mudança de pose e oclusão no fim do trecho, confirmando que início e término da ação afetam a confiança, "
                "mesmo quando a classe não muda.",
                body,
            ),
            paragraph("Parecer dos integrantes", heading),
        ]
    )
    opinions = [
        ["Guilherme", "A integração é reproduzível; rastreamento é a melhoria prioritária."],
        ["Enricco", "O YOLO foi estável, mas cenas com várias pessoas exigem identidade temporal."],
        ["Gabriel", "A máscara foi boa em luz natural e falhou no palco escuro, evidenciando domínio."],
        ["Danilo", "O SlowFast acertou as ações; a confiança refletiu as transições do movimento."],
        ["Laura", "A sobreposição e os arquivos tabulares tornam os erros fáceis de demonstrar."],
    ]
    opinion_table = Table(opinions, colWidths=[2.8 * cm, 14.6 * cm])
    opinion_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), FONT_BOLD),
                ("FONTNAME", (1, 0), (1, -1), FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9CBD8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            opinion_table,
            Spacer(1, 0.28 * cm),
            paragraph("Conclusão", heading),
            paragraph(
                "O pipeline cumpriu a integração solicitada e gerou vídeo, caixa, máscara, ação e confiança sem treinamento adicional. O experimento mostrou "
                "alto desempenho no cenário externo e expôs duas limitações importantes: associação do alvo em cenas com várias pessoas e segmentação em "
                "baixa iluminação. Os resultados ruins foram preservados porque explicam onde cada modelo precisa de contexto, rastreamento ou adaptação de domínio.",
                body,
            ),
            Spacer(1, 0.3 * cm),
            Table(
                [[paragraph("<font color='#FFFFFF'>Código, configurações, testes, vídeo demonstrativo e evidências estão versionados no repositório do grupo.</font>", caption)]],
                colWidths=[17.4 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), BLUE),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                        ("TOPPADDING", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                    ]
                ),
            ),
        ]
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


if __name__ == "__main__":
    build()
