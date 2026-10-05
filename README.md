# CP5 - Deteccao de Acoes com YOLO, U-Net e SlowFast

![Arquitetura do pipeline](docs/images/architecture.svg)

Projeto da disciplina **Applied Computer Vision (2026)**. A aplicacao le videos curtos e combina tres modelos pre-treinados, sem treinamento adicional, para mostrar:

- a caixa da pessoa detectada pelo **YOLO**;
- os pixels da pessoa segmentados por uma **U-Net**;
- a acao e a confianca estimadas pelo **SlowFast** em janelas temporais sucessivas.

O processamento gera um MP4 anotado, um CSV com os dados de cada quadro e um JSON com as predicoes de cada janela. O modo opcional `--person-crop` executa o SlowFast apenas no recorte da pessoa para permitir a comparacao solicitada como extensao.

## Entrega pronta

- [Relatorio final em PDF - 3 paginas](docs/relatorio_cp5.pdf)
- [Video demonstrativo com as tres saidas sobrepostas](docs/demo_cp5.mp4)
- [Protocolo de experimentos](docs/PROTOCOLO_EXPERIMENTOS.md)

| Video 1 - arco e flecha | Video 2 - danca em palco |
|---|---|
| ![YOLO, mascara e archery](docs/images/evidencias/video_1_archery_anotado_momento_2.jpg) | ![YOLO e dancing ballet no palco](docs/images/evidencias/video_2_theatre_anotado_momento_2.jpg) |

Resultados medidos: o YOLO detectou pessoa em 64/64 quadros nos dois videos. O SlowFast classificou `archery` com 100% nas tres janelas do primeiro e `dancing ballet` com 88,8%, 83,7% e 75,0% no segundo. Em baixa iluminacao, a U-Net gerou mascara nao vazia em apenas 12/64 quadros; a falha foi mantida e explicada no relatorio.

## Integrantes

| Nome | RM | Responsabilidade principal |
|---|---:|---|
| Guilherme Santos | 551168 | integracao, testes e documentacao |
| Enricco | 551717 | YOLO e leitura dos videos |
| Gabriel | 99227 | U-Net e projecao da mascara |
| Danilo | 99465 | SlowFast e janelas temporais |
| Laura | 98747 | experimentos, analise e apresentacao |

Todos os integrantes participam da analise dos resultados e do parecer final.

## Requisitos atendidos

| Item do CP5 | Implementacao |
|---|---|
| YOLO para pessoa | `YoloPersonDetector`, classe COCO `person` |
| U-Net no recorte | U2-Net Human Segmentation por padrao ou checkpoint TorchScript do Teams |
| Mascara no quadro original | `project_mask`, com recorte expandido e reprojecao |
| SlowFast em janelas | SlowFast R50/Kinetics-400, 32 quadros e stride 16 |
| Acao e confianca | sobreposicao no video e JSON por janela |
| Dois videos e tres momentos | protocolo e extrator automatico de evidencias |
| Casos de falha | roteiro de oclusao, transicao, movimento e multiplas pessoas |
| Extensao opcional | `--person-crop` |
| Relatorio de ate 3 paginas | `docs/relatorio_cp5.pdf` |

## Instalacao

Recomendado: Python 3.11, Git, FFmpeg e pelo menos 8 GB de RAM. GPU CUDA e opcional, mas acelera muito o SlowFast.

```bash
git clone https://github.com/GuilhermeSSantos2004/Slow-Fast.git
cd Slow-Fast
python -m venv .venv
```

No Windows:

```powershell
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

No Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Na primeira execucao, os pesos pre-treinados do YOLO, U2-Net Human e SlowFast sao baixados automaticamente. Nenhum modelo e treinado por este projeto.

## Execucao

Coloque dois videos em `assets/videos/` e execute:

```bash
python -m action_vision \
  assets/videos/video_1.mp4 \
  assets/videos/video_2.mp4 \
  --config configs/default.yaml \
  --output outputs
```

Para a extensao com o recorte da pessoa:

```bash
python -m action_vision assets/videos/video_1.mp4 --person-crop --output outputs/recorte
```

Depois, extraia tres evidencias de cada resultado:

```bash
python scripts/extract_evidence.py \
  outputs/video_1_anotado.mp4 \
  outputs/video_2_anotado.mp4
```

### Usar a U-Net fornecida no Teams

O backend padrao e a **U2-Net Human Segmentation**, arquitetura aninhada da familia U-Net ja treinada para pessoas. Se o professor exigir exatamente o arquivo disponibilizado no Teams, exporte-o como TorchScript e altere o YAML:

```yaml
unet:
  backend: torchscript
  checkpoint: models/unet_pessoa.torchscript.pt
  threshold: 0.50
```

O modelo deve receber tensor RGB `N x 3 x H x W`, normalizado em `[0, 1]`, e devolver logits de mascara. Assim o restante do pipeline nao muda.

## Resultados gerados

Para cada entrada `nome.mp4`, sao criados:

| Arquivo | Conteudo |
|---|---|
| `nome_anotado.mp4` | caixa, mascara, acao e confianca sobrepostas |
| `nome_frames.csv` | tempo, YOLO, area da mascara e ultima acao por quadro |
| `nome_janelas.json` | classe e confianca por janela SlowFast |

Videos e pesos nao sao enviados ao Git por padrao. As evidencias selecionadas e o relatorio final ficam em `docs/`.

## Testes e qualidade

```bash
python -m pytest -q
python -m ruff check src tests scripts
```

Os testes cobrem validacao da configuracao, escolha da pessoa principal, reprojecao espacial da mascara, recorte temporal e cobertura da ultima janela. O GitHub Actions repete essas verificacoes em todo push e pull request.

## Decisoes tecnicas e limitacoes

- Apenas a deteccao principal e segmentada. Em cenas com varias pessoas, a selecao pondera confianca e area; isso reduz alternancia, mas nao substitui rastreamento.
- A mesma janela SlowFast pode conter o fim de uma acao e o inicio de outra. Nessa transicao, a confianca tende a cair ou a classe pode mudar.
- Kinetics-400 limita as classes possiveis. Uma acao fora desse vocabulario sera aproximada para alguma classe conhecida.
- O modo de quadro inteiro preserva contexto; o recorte reduz distracoes, mas pode remover objetos importantes para reconhecer a acao.
- O custo computacional principal e o SlowFast. Em CPU, use videos curtos; em GPU, mantenha `device: auto`.

O protocolo completo para comparar os dois videos esta em [docs/PROTOCOLO_EXPERIMENTOS.md](docs/PROTOCOLO_EXPERIMENTOS.md).

## Fontes dos modelos

- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [U2-Net: Going Deeper with Nested U-Structure](https://arxiv.org/abs/2005.09007)
- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)
- [PyTorchVideo](https://pytorchvideo.org/)
- [Kinetics-400](https://www.deepmind.com/open-source/kinetics)

## Privacidade

Use apenas videos autorizados. Evite publicar rostos ou locais privados sem consentimento. Resultados de modelos pre-treinados podem conter vieses e devem ser tratados como estimativas, nao como identificacao de pessoas ou prova de comportamento.
