"""
Gera relatório PDF do projeto MobileNet com gráficos e insights.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg
from datetime import date

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_EDA = os.path.join(BASE, 'images', 'eda')
IMAGES_TRN = os.path.join(BASE, 'images', 'training')
IMAGES_EVA = os.path.join(BASE, 'images', 'evaluation')
OUTPUT     = os.path.join(BASE, 'reports', 'relatorio_mobilenet.pdf')

AZUL      = '#1B4F72'
AZUL_CLR  = '#2E86C1'
CINZA     = '#566573'
BRANCO    = '#FFFFFF'
FUNDO     = '#F2F3F4'

def sem_eixos(ax):
    ax.axis('off')

def titulo_secao(ax, texto, subtexto=''):
    sem_eixos(ax)
    ax.set_facecolor(FUNDO)
    ax.add_patch(FancyBboxPatch((0.03, 0.1), 0.94, 0.8,
                                boxstyle='round,pad=0.02',
                                facecolor=AZUL, edgecolor='none',
                                transform=ax.transAxes, zorder=0))
    ax.text(0.5, 0.58, texto, ha='center', va='center',
            fontsize=18, fontweight='bold', color=BRANCO,
            transform=ax.transAxes)
    if subtexto:
        ax.text(0.5, 0.3, subtexto, ha='center', va='center',
                fontsize=11, color='#AED6F1', transform=ax.transAxes)

def insight_box(ax, texto, cor_fundo='#EAF2FF', cor_borda=AZUL_CLR):
    sem_eixos(ax)
    ax.add_patch(FancyBboxPatch((0.01, 0.05), 0.98, 0.9,
                                boxstyle='round,pad=0.02',
                                facecolor=cor_fundo, edgecolor=cor_borda,
                                linewidth=1.5,
                                transform=ax.transAxes))
    ax.text(0.5, 0.5, texto, ha='center', va='center',
            fontsize=9.5, color='#1A252F', transform=ax.transAxes,
            wrap=True, multialignment='left',
            bbox=dict(facecolor='none', edgecolor='none', pad=8))

def embed_img(ax, caminho):
    sem_eixos(ax)
    if os.path.exists(caminho):
        img = mpimg.imread(caminho)
        ax.imshow(img, aspect='auto')

with PdfPages(OUTPUT) as pdf:

    # ── CAPA ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(AZUL)
    ax = fig.add_axes([0, 0, 1, 1])
    sem_eixos(ax)

    ax.add_patch(FancyBboxPatch((0.05, 0.08), 0.9, 0.84,
                                boxstyle='round,pad=0.02',
                                facecolor='#154360', edgecolor='#AED6F1',
                                linewidth=2, transform=ax.transAxes))

    ax.text(0.5, 0.78, 'MobileNet', ha='center', va='center',
            fontsize=42, fontweight='bold', color=BRANCO,
            transform=ax.transAxes, family='monospace')

    ax.text(0.5, 0.65, 'Classificação de Faixa de Preço de Celulares\ncom Rede Neural MLP',
            ha='center', va='center', fontsize=16, color='#AED6F1',
            transform=ax.transAxes, linespacing=1.6)

    ax.plot([0.15, 0.85], [0.52, 0.52], color='#AED6F1', linewidth=0.8,
            transform=ax.transAxes)

    resumo = (
        'Pipeline completo: EDA  →  Implementação MLP  →  Avaliação  →  '
        'Ajuste de Hiperparâmetros  →  Resultados Finais'
    )
    ax.text(0.5, 0.45, resumo, ha='center', va='center',
            fontsize=10, color='#D6EAF8', transform=ax.transAxes,
            style='italic')

    metricas = '95,5% de acurácia  ·  F1-macro 95,51%  ·  20 features  ·  2 000 amostras'
    ax.text(0.5, 0.32, metricas, ha='center', va='center',
            fontsize=12, color=BRANCO, fontweight='bold',
            transform=ax.transAxes)

    ax.text(0.5, 0.16, f'Autor: Filipe Rangel  ·  {date.today().strftime("%B de %Y")}',
            ha='center', va='center', fontsize=10, color='#85C1E9',
            transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PÁG 2: VISÃO GERAL DO PROJETO ────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(FUNDO)
    gs  = gridspec.GridSpec(6, 2, figure=fig,
                            hspace=0.55, wspace=0.35,
                            left=0.06, right=0.94,
                            top=0.93, bottom=0.05)

    ax_tit = fig.add_subplot(gs[0, :])
    titulo_secao(ax_tit, '01 — Visão Geral do Projeto', 'Problema · Dataset · Pipeline')

    # Problema
    ax_p = fig.add_subplot(gs[1:3, 0])
    sem_eixos(ax_p)
    ax_p.set_facecolor(BRANCO)
    ax_p.add_patch(FancyBboxPatch((0,0), 1, 1, boxstyle='round,pad=0.02',
                                  facecolor=BRANCO, edgecolor='#BDC3C7',
                                  transform=ax_p.transAxes))
    problema = (
        'PROBLEMA\n\n'
        'Dado um conjunto de 20 especificações\n'
        'técnicas de celulares, prever em qual das\n'
        '4 faixas de preço o aparelho se enquadra:\n\n'
        '  Classe 0 — Baixo custo\n'
        '  Classe 1 — Custo médio\n'
        '  Classe 2 — Alto custo\n'
        '  Classe 3 — Custo muito alto'
    )
    ax_p.text(0.5, 0.5, problema, ha='center', va='center',
              fontsize=9.5, transform=ax_p.transAxes,
              linespacing=1.6, color='#1A252F')

    # Dataset
    ax_d = fig.add_subplot(gs[1:3, 1])
    sem_eixos(ax_d)
    ax_d.add_patch(FancyBboxPatch((0,0), 1, 1, boxstyle='round,pad=0.02',
                                  facecolor=BRANCO, edgecolor='#BDC3C7',
                                  transform=ax_d.transAxes))
    dataset = (
        'DATASET\n\n'
        '  2 000 amostras de treino\n'
        '  1 000 amostras de teste\n'
        '  20 features numéricas\n'
        '  4 classes — 500 amostras cada\n\n'
        'Features principais:\n'
        'RAM · Potência da bateria ·\n'
        'Resolução · Câmeras · Memória'
    )
    ax_d.text(0.5, 0.5, dataset, ha='center', va='center',
              fontsize=9.5, transform=ax_d.transAxes,
              linespacing=1.6, color='#1A252F')

    # Pipeline
    ax_pipe = fig.add_subplot(gs[3:, :])
    sem_eixos(ax_pipe)
    etapas = [
        ('EDA', '#1A5276'),
        ('Normalização\nStandardScaler', '#1F618D'),
        ('MLP\n128→64', '#2471A3'),
        ('Avaliação\n92% acc', '#2E86C1'),
        ('Tuning\nGridSearch', '#3498DB'),
        ('Resultado Final\n95,5% acc', '#5DADE2'),
    ]
    n = len(etapas)
    for i, (nome, cor) in enumerate(etapas):
        x = 0.04 + i * (0.92 / n)
        w = 0.92 / n - 0.02
        ax_pipe.add_patch(FancyBboxPatch((x, 0.2), w, 0.6,
                                         boxstyle='round,pad=0.01',
                                         facecolor=cor, edgecolor='none',
                                         transform=ax_pipe.transAxes))
        ax_pipe.text(x + w/2, 0.5, nome, ha='center', va='center',
                     fontsize=8.5, color=BRANCO, fontweight='bold',
                     transform=ax_pipe.transAxes)
        if i < n - 1:
            ax_pipe.annotate('', xy=(x + w + 0.015, 0.5),
                             xytext=(x + w + 0.003, 0.5),
                             xycoords='axes fraction',
                             arrowprops=dict(arrowstyle='->', color=AZUL, lw=1.5))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PÁG 3: EDA ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(FUNDO)
    gs  = gridspec.GridSpec(5, 2, figure=fig,
                            hspace=0.5, wspace=0.3,
                            left=0.05, right=0.95,
                            top=0.93, bottom=0.05)

    ax_tit = fig.add_subplot(gs[0, :])
    titulo_secao(ax_tit, '02 — Análise Exploratória (EDA)',
                 'Distribuição do alvo · Correlação · Top Features')

    ax_img1 = fig.add_subplot(gs[1:3, 0])
    embed_img(ax_img1, os.path.join(IMAGES_EDA, '01_target_distribution.png'))

    ax_ins1 = fig.add_subplot(gs[3:, 0])
    insight_box(ax_ins1,
        'Dataset perfeitamente balanceado: 500 amostras por classe (25% cada).\n'
        'Elimina viés do modelo e torna a acurácia uma métrica confiável.')

    ax_img2 = fig.add_subplot(gs[1:3, 1])
    embed_img(ax_img2, os.path.join(IMAGES_EDA, '02_correlation_heatmap.png'))

    ax_ins2 = fig.add_subplot(gs[3:, 1])
    insight_box(ax_ins2,
        'RAM é a feature mais correlacionada com faixa_preco (r ≈ 0,92).\n'
        'Baixa multicolinearidade entre features — cada variável traz\n'
        'informação independente ao modelo.')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PÁG 4: BOXPLOTS ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(FUNDO)
    gs  = gridspec.GridSpec(5, 1, figure=fig,
                            hspace=0.4,
                            left=0.05, right=0.95,
                            top=0.93, bottom=0.05)

    ax_tit = fig.add_subplot(gs[0])
    titulo_secao(ax_tit, '02 — EDA: Distribuição das Top 6 Features por Classe')

    ax_img = fig.add_subplot(gs[1:4])
    embed_img(ax_img, os.path.join(IMAGES_EDA, '03_features_boxplots.png'))

    ax_ins = fig.add_subplot(gs[4])
    insight_box(ax_ins,
        'RAM: separação quase perfeita entre classes (~750 / ~1600 / ~2600 / ~3500 MB). '
        'Feature mais discriminativa do modelo.\n'
        'Potência da bateria e resolução: tendência crescente com o preço, mas com sobreposição entre classes adjacentes.\n'
        'Memória interna e largura da tela: distribuições similares entre classes — baixo poder discriminativo isolado.')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PÁG 5: ARQUITETURA E TREINAMENTO ─────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(FUNDO)
    gs  = gridspec.GridSpec(6, 2, figure=fig,
                            hspace=0.55, wspace=0.3,
                            left=0.05, right=0.95,
                            top=0.93, bottom=0.05)

    ax_tit = fig.add_subplot(gs[0, :])
    titulo_secao(ax_tit, '03 — Arquitetura MLP e Treinamento',
                 'Arquitetura baseline · Curva de aprendizado')

    # Diagrama da arquitetura
    ax_arq = fig.add_subplot(gs[1:4, 0])
    sem_eixos(ax_arq)
    camadas = [
        ('Entrada\n20 features', 20, '#1A5276'),
        ('Oculta 1\n128 neurônios\nReLU', 10, '#2471A3'),
        ('Oculta 2\n64 neurônios\nReLU', 7,  '#2E86C1'),
        ('Saída\n4 classes\nSoftmax', 4,  '#5DADE2'),
    ]
    xs = [0.12, 0.38, 0.63, 0.88]
    for (nome, n, cor), x in zip(camadas, xs):
        ax_arq.add_patch(FancyBboxPatch((x-0.1, 0.1), 0.2, 0.8,
                                        boxstyle='round,pad=0.02',
                                        facecolor=cor, edgecolor='none',
                                        transform=ax_arq.transAxes))
        ax_arq.text(x, 0.5, nome, ha='center', va='center',
                    fontsize=8, color=BRANCO, fontweight='bold',
                    transform=ax_arq.transAxes)
        if x != xs[-1]:
            ax_arq.annotate('', xy=(x+0.12, 0.5), xytext=(x+0.1, 0.5),
                            xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color=AZUL, lw=1.5))

    ax_ins_arq = fig.add_subplot(gs[4:, 0])
    insight_box(ax_ins_arq,
        'Arquitetura baseline: 20 → 128 → 64 → 4\n'
        'Ativação: ReLU nas ocultas, Softmax na saída\n'
        'Otimizador: Adam · Early stopping (paciência=20)\n'
        'Função de perda: Entropia Cruzada')

    ax_img_lc = fig.add_subplot(gs[1:4, 1])
    embed_img(ax_img_lc, os.path.join(IMAGES_TRN, '01_learning_curve.png'))

    ax_ins_lc = fig.add_subplot(gs[4:, 1])
    insight_box(ax_ins_lc,
        'Convergência rápida nas primeiras 10 iterações.\n'
        'Perda de treino → ~0,03 | validação estabiliza em ~0,12.\n'
        'Early stopping interrompeu em ~53 iterações.\n'
        'Gap aceitável — overfitting leve e controlado.')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PÁG 6: AVALIAÇÃO ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(FUNDO)
    gs  = gridspec.GridSpec(6, 2, figure=fig,
                            hspace=0.55, wspace=0.3,
                            left=0.05, right=0.95,
                            top=0.93, bottom=0.05)

    ax_tit = fig.add_subplot(gs[0, :])
    titulo_secao(ax_tit, '04 — Avaliação do Modelo Baseline',
                 'Matriz de confusão · Métricas por classe')

    ax_img1 = fig.add_subplot(gs[1:4, 0])
    embed_img(ax_img1, os.path.join(IMAGES_EVA, '01_confusion_matrix.png'))

    ax_ins1 = fig.add_subplot(gs[4:, 0])
    insight_box(ax_ins1,
        '368 / 400 amostras corretas → 92% de acurácia.\n'
        'Todos os erros ocorrem entre classes adjacentes.\n'
        'Nenhum salto extremo (Baixo → Muito alto).\n'
        'Alto custo é a classe mais difícil (13 erros).')

    ax_img2 = fig.add_subplot(gs[1:4, 1])
    embed_img(ax_img2, os.path.join(IMAGES_EVA, '02_metrics_per_class.png'))

    ax_ins2 = fig.add_subplot(gs[4:, 1])
    insight_box(ax_ins2,
        'Classes extremas (Baixo / Muito alto): F1 = 0,95\n'
        'Classes do meio (Médio / Alto): F1 = 0,89\n'
        'Padrão esperado: classes intermediárias têm\n'
        'maior ambiguidade com seus vizinhos.')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PÁG 7: HIPERPARÂMETROS ───────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(FUNDO)
    gs  = gridspec.GridSpec(6, 2, figure=fig,
                            hspace=0.55, wspace=0.3,
                            left=0.05, right=0.95,
                            top=0.93, bottom=0.05)

    ax_tit = fig.add_subplot(gs[0, :])
    titulo_secao(ax_tit, '05 — Ajuste de Hiperparâmetros',
                 'RandomizedSearchCV (40 combinações) → GridSearchCV refinado')

    ax_img1 = fig.add_subplot(gs[1:4, 0])
    embed_img(ax_img1, os.path.join(IMAGES_EVA, '03_hyperparam_search.png'))

    ax_ins1 = fig.add_subplot(gs[4:, 0])
    insight_box(ax_ins1,
        '40 combinações testadas com CV 3-fold.\n'
        'Top-20 combinações superaram o baseline (0,9198).\n'
        'Melhor F1-macro em CV: 0,9511\n'
        'Estratégia em 2 etapas: busca ampla → refinamento.')

    ax_img2 = fig.add_subplot(gs[1:4, 1])
    embed_img(ax_img2, os.path.join(IMAGES_EVA, '04_model_comparison.png'))

    ax_ins2 = fig.add_subplot(gs[4:, 1])
    insight_box(ax_ins2,
        'Configuração vencedora: 256→128 · tanh · α=0,1\n\n'
        'Baseline  →  92,00% acc | 91,98% F1\n'
        'Tuned      →  95,50% acc | 95,51% F1\n\n'
        'Ganho: +3,5 p.p. acurácia / +3,53 p.p. F1-macro')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PÁG 8: RESULTADOS FINAIS ─────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(FUNDO)
    gs  = gridspec.GridSpec(6, 2, figure=fig,
                            hspace=0.55, wspace=0.3,
                            left=0.05, right=0.95,
                            top=0.93, bottom=0.05)

    ax_tit = fig.add_subplot(gs[0, :])
    titulo_secao(ax_tit, '06 — Resultados Finais',
                 'Importância das features · Predições no conjunto de teste')

    ax_img1 = fig.add_subplot(gs[1:4, 0])
    embed_img(ax_img1, os.path.join(IMAGES_EVA, '06_feature_importance.png'))

    ax_ins1 = fig.add_subplot(gs[4:, 0])
    insight_box(ax_ins1,
        'RAM causa queda de ~68 p.p. de acurácia ao ser permutada.\n'
        'Potência da bateria: ~21 p.p.  |  Resolução: ~10 p.p.\n'
        'As outras 16 features têm importância próxima de zero.\n'
        '4 features explicam quase toda a capacidade preditiva.')

    ax_img2 = fig.add_subplot(gs[1:4, 1])
    embed_img(ax_img2, os.path.join(IMAGES_EVA, '05_test_predictions.png'))

    ax_ins2 = fig.add_subplot(gs[4:, 1])
    insight_box(ax_ins2,
        'Predições no teste: 256 (Baixo) · 227 (Médio) ·\n'
        '258 (Alto) · 259 (Muito alto)\n'
        'Distribuição equilibrada e coerente com o treino.\n'
        'Nenhum viés sistemático em direção a uma classe.')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ── PÁG 9: CONCLUSÕES ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(FUNDO)
    gs  = gridspec.GridSpec(7, 2, figure=fig,
                            hspace=0.6, wspace=0.35,
                            left=0.06, right=0.94,
                            top=0.93, bottom=0.05)

    ax_tit = fig.add_subplot(gs[0, :])
    titulo_secao(ax_tit, '07 — Conclusões e Aprendizados')

    # Tabela de resumo
    ax_tab = fig.add_subplot(gs[1:4, :])
    sem_eixos(ax_tab)
    linhas = [
        ['Dataset',               '2 000 amostras · 20 features · 4 classes balanceadas'],
        ['Normalização',          'StandardScaler (μ=0, σ=1)'],
        ['Arquitetura final',     '20 → 256 → 128 → 4  (tanh + Softmax)'],
        ['Otimizador',            'Adam  ·  lr = 0,001  ·  Early stopping'],
        ['Regularização L2',      'α = 0,1  (reduz overfitting do baseline)'],
        ['Acurácia — baseline',   '92,00%'],
        ['Acurácia — modelo final','95,50%  (+3,5 p.p.)'],
        ['F1-macro — modelo final','95,51%  (+3,53 p.p.)'],
        ['Feature mais importante','RAM  (queda de ~68 p.p. ao permutar)'],
    ]
    tabela = ax_tab.table(
        cellText=linhas,
        colLabels=['Parâmetro / Etapa', 'Resultado'],
        loc='center',
        cellLoc='left',
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(9)
    tabela.scale(1, 1.55)
    for (r, c), cell in tabela.get_celld().items():
        cell.set_edgecolor('#BDC3C7')
        if r == 0:
            cell.set_facecolor(AZUL)
            cell.set_text_props(color=BRANCO, fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#EAF2FF')
        else:
            cell.set_facecolor(BRANCO)
        if c == 0:
            cell.set_text_props(fontweight='bold', color=AZUL)

    # Aprendizados
    aprendizados = [
        ('Dados balanceados simplificam a avaliação',
         'Com 25% por classe, a acurácia é métrica suficiente\ne não é enganada por classes majoritárias.'),
        ('RAM domina a predição',
         'Uma única feature explica ~68% da queda de acurácia\nao ser removida — feature engineering poderia explorá-la.'),
        ('Tuning trouxe ganho real',
         'Arquitetura maior (256→128) + tanh + L2 forte\n(α=0,1) superaram o baseline em 3,5 p.p.'),
        ('Erros são estruturalmente coerentes',
         'Todas as confusões ocorrem entre classes adjacentes:\no modelo respeita a ordem ordinal do preço.'),
    ]
    for i, (titulo, detalhe) in enumerate(aprendizados):
        col = i % 2
        row = 4 + (i // 2)
        ax_ap = fig.add_subplot(gs[row, col])
        sem_eixos(ax_ap)
        ax_ap.add_patch(FancyBboxPatch((0.01, 0.05), 0.98, 0.9,
                                       boxstyle='round,pad=0.02',
                                       facecolor='#EAF2FF', edgecolor=AZUL_CLR,
                                       linewidth=1.2, transform=ax_ap.transAxes))
        ax_ap.text(0.5, 0.72, titulo, ha='center', va='center',
                   fontsize=9, fontweight='bold', color=AZUL,
                   transform=ax_ap.transAxes)
        ax_ap.text(0.5, 0.35, detalhe, ha='center', va='center',
                   fontsize=8.3, color=CINZA, transform=ax_ap.transAxes,
                   linespacing=1.5)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f'Relatório gerado: {OUTPUT}')
