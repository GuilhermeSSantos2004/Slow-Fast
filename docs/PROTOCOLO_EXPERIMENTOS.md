# Protocolo de experimentos

Este protocolo impede que a avaliacao dependa apenas de impressoes visuais e garante que os dois videos sejam analisados do mesmo modo.

## Videos e momentos

Use dois videos curtos com uma pessoa visivel e a acao completa. Em cada video, registre no minimo tres momentos:

1. inicio da acao;
2. execucao estavel;
3. termino ou transicao.

Nao use o mesmo trecho renomeado. Guarde os originais em `assets/videos/` apenas localmente, respeitando privacidade e direitos de uso.

## Criterios de avaliacao

| Saida | Pergunta | Registro sugerido |
|---|---|---|
| YOLO | A caixa acompanha a pessoa? | sim/parcial/nao e confianca |
| U-Net | A mascara acompanha o contorno? | sim/parcial/nao e artefatos observados |
| SlowFast | A classe corresponde ao trecho? | classe, confianca e classe esperada |
| Transicao | A confianca cai ou a classe muda? | comparar janelas antes/durante/depois |

Os CSVs por quadro e os JSONs por janela ficam em `outputs/`. Use `scripts/extract_evidence.py` para exportar os quadros escolhidos com nomes reproduziveis.

## Falhas que devem ser procuradas

- oclusao parcial, pessoa pequena ou fora do centro;
- movimento rapido e borrado;
- fundo com objetos visualmente semelhantes;
- mais de uma pessoa no quadro;
- inicio ou fim da acao dentro de uma mesma janela temporal;
- acao fora ou ambigua nas 400 classes do Kinetics.

Explique pelo menos dois erros no relatorio. Nao remova resultados ruins: eles fazem parte da analise pedida.

