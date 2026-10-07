# Protocolo de experimentos CP5

## Entradas reproduzíveis

| Vídeo | Arquivo público | Trecho original | Entrada utilizada |
|---|---|---|---|
| 1 | `archery.mp4`, PyTorchVideo | 0–10 s | 10 s, 8 FPS, 320×240 |
| 2 | `theatre.webm`, PyTorchVideo / Theatre Flamenco of San Francisco | 1–11 s | 10 s, 8 FPS, 320×180 |

Os trechos estão versionados em `assets/videos/`. As URLs e os hashes dos originais estão em `scripts/prepare_videos.py` e `docs/results/reproducao.json`. A resolução mantém a proporção de cada fonte.

## Execução

1. Instalar o ambiente de `requirements-cpu.txt` e FFmpeg.
2. Executar `python scripts/reproduce_cp5.py`.
3. Conferir os três modelos isolados, a integração e os CSV/JSON.
4. Inspecionar os quadros 0, 40 e 79 (0 s, 5 s e 9,875 s).
5. Executar `python scripts/check_delivery.py`.

A U-Net isolada recebe o quadro inteiro. A integração executa U-Net nos recortes YOLO e prioriza candidatos segmentáveis; portanto, não se assume que as duas máscaras sejam idênticas. O SlowFast é executado sem YOLO/U-Net para a classificação do quadro inteiro. A extensão usa caixas do CSV integrado para comparar o recorte das mesmas pessoas.

## Inspeção visual

| Saída | Pergunta | Como registrar |
|---|---|---|
| Caixa | Acompanha a pessoa escolhida e a ação? | sim/parcial/não, coordenadas e imagem |
| Máscara | Acompanha o contorno? | pixels não vazios, partes perdidas e fundo incluído |
| Ação | Corresponde à cena? | classe prevista, confiança e descrição visual |
| Transição | Classe/confiança mudam no início ou fim? | fases observadas e comparação das quatro janelas |

As observações dos seis momentos estão em `docs/results/avaliacao_manual.json`. Elas descrevem a execução verificada; devem ser revistas se a configuração ou as entradas mudarem. Sem máscaras verdadeiras não se calcula IoU/Dice, e contagem de detecções não é acurácia.

## Resultado observado

- Arco e flecha: caixa acompanha o arqueiro selecionado; contorno parcial; `archery` nas quatro janelas com confiança de 100%.
- Palco: associação do alvo e segmentação falham em parte do trecho. A máscara integrada é não vazia em 33/80 quadros, contra 79/80 na U-Net isolada.
- A cena é flamenco. As previsões `tango dancing` aproximam dança, mas não o gênero. `giving or receiving award`, na janela 4–8 s, é um erro de ação.
- As fases preparação, mira e recuperação estão presentes no vídeo 1. A confiança não cai e a classe não muda na janela final. O trecho do vídeo 2 não contém início/término completo da dança; alterações de previsão não provam uma transição real.

## Falhas explicadas no PDF

1. Seleção de participante ao fundo e máscara ausente/incompleta: comparar resultado integrado e U-Net isolada; não atribuir a falha somente à iluminação.
2. Classe de premiação em uma cena de dança: preservar a previsão incorreta e discutir contexto, pose e vocabulário como hipóteses, sem afirmar causa comprovada.

O relatório tem três páginas, tabela dos seis momentos, imagens, duas falhas, análise temporal, comparação opcional e parecer técnico por integrante. Todos os participantes devem revisar a análise antes do envio.
