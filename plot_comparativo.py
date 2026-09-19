import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

# ==============================================================================
# BASE DE DADOS CONSOLIDADA DA LITERATURA (CEC 2014)
# ==============================================================================
LITERATURA = {
    30: {  # D = 30
        2:  {'L-SHADE (Camp.)': {'best': 0.0, 'mean': 0.0, 'std': 0.0}},
        4:  {'L-SHADE (Camp.)': {'best': 0.0, 'mean': 0.0, 'std': 0.0}}, # PREENCHER MEAN/STD OLHANDO A TABELA I DO PDF
        6:  {'L-SHADE (Camp.)': {'best': 0.0, 'mean': 1.40e-7, 'std': 9.90e-7}}, # PREENCHER MEAN/STD OLHANDO A TABELA I DO PDF
        7:  {'L-SHADE (Camp.)': {'best': 0.0, 'mean': 0.0, 'std': 0.0}}, # PREENCHER MEAN/STD OLHANDO A TABELA I DO PDF
        9:  {'L-SHADE (Camp.)': {'best': 3.30e+0, 'mean': 6.80e+0, 'std': 5.40e+0}}, # PREENCHER MEAN/STD OLHANDO A TABELA I DO PDF
        13: {'L-SHADE (Camp.)': {'best': 9.00e-2, 'mean': 1.20e-1, 'std': 1.70e-2}} # PREENCHER MEAN/STD OLHANDO A TABELA I DO PDF
    },
    50: {  # D = 50
        2:  {'L-SHADE (Camp.)': {'best': 0.0, 'mean': 0.0, 'std': 0.0}},
        4:  {'L-SHADE (Camp.)': {'best': 4.20e-1, 'mean': 5.90e+1, 'std': 4.60e+1}}, # PREENCHER MEAN/STD OLHANDO A TABELA I DO PDF
        6:  {'L-SHADE (Camp.)': {'best': 4.9e-5, 'mean': 2.60e-1, 'std': 5.2e-1}}, # PREENCHER MEAN/STD OLHANDO A TABELA I DO PDF
        7:  {'L-SHADE (Camp.)': {'best': 0.0, 'mean': 0.0, 'std': 0.0}}, # PREENCHER MEAN/STD OLHANDO A TABELA I DO PDF
        9:  {'L-SHADE (Camp.)': {'best': 5.4e+0, 'mean': 1.1e+1, 'std': 2.1e+0}}, # PREENCHER MEAN/STD OLHANDO A TABELA I DO PDF
        13: {'L-SHADE (Camp.)': {'best': 1.1e-1, 'mean': 1.6e-1, 'std': 1.8e-2}} # PREENCHER MEAN/STD OLHANDO A TABELA I DO PDF
    }
}

def plotar_comparativo_completo(funcao, dimensao, dados_usuario, dados_literatura):
    # Aumento do tamanho da fonte e proporção mais quadrada (8x6) ideal para IEEE
    plt.rcParams.update({'font.size': 13, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    # 1. Preparar os dados do usuário
    df_list = []
    nomes_usuario = list(dados_usuario.keys())
    for alg in nomes_usuario:
        erros = dados_usuario[alg]
        erros_seguros = np.where(np.array(erros) < 1e-8, 1e-8, np.array(erros))
        df_list.append(pd.DataFrame({'Algoritmo': alg, 'Erro': erros_seguros}))
        
    df_user = pd.concat(df_list, ignore_index=True)
    num_user_algs = len(nomes_usuario)
    
    # 2. Plota os Boxplots dos modelos avaliados
    paleta = sns.color_palette("Set2", num_user_algs)
    sns.boxplot(
        x='Algoritmo', y='Erro', data=df_user, ax=ax,
        palette=paleta, width=0.5,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 9}
    )
    
    # Destaca o Melhor Valor dos modelos avaliados
    for i, alg in enumerate(nomes_usuario):
        erros = dados_usuario[alg]
        melhor_real = np.min(erros)
        y_plot = max(melhor_real, 1e-8)
        
        ax.scatter(i, y_plot, color='gold', edgecolor='black', s=220, zorder=10, marker='*')
        texto_valor = "0.00e+00" if melhor_real < 1e-8 else f"{melhor_real:.2e}"
        ax.annotate(texto_valor, xy=(i, y_plot), xytext=(12, 0), textcoords="offset points",
                    va='center', ha='left', fontsize=10, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.9))

    # 3. Plota os dados da Literatura
    nomes_lit = list(dados_literatura.keys())
    posicoes_lit = np.arange(num_user_algs, num_user_algs + len(nomes_lit))
    
    for i, (alg, stats) in enumerate(dados_literatura.items()):
        pos = posicoes_lit[i]
        mean_val = max(stats['mean'], 1e-8)
        std_val = stats['std']
        best_val = stats['best']
        
        yerr_lower = mean_val - max(mean_val - std_val, 1e-8)
        yerr_upper = std_val
        
        ax.errorbar(x=pos, y=mean_val, yerr=[[yerr_lower], [yerr_upper]],
                    fmt='s', color='#2b2b2b', ecolor='#4a4a4a',
                    elinewidth=2.0, capsize=6, capthick=2.0, markersize=8, zorder=8)
        
        y_best_plot = max(best_val, 1e-8)
        ax.scatter(pos, y_best_plot, color='gold', edgecolor='black', s=220, zorder=10, marker='*')
        
        texto_best = "0.00e+00" if best_val < 1e-8 else f"{best_val:.2e}"
        ax.annotate(texto_best, xy=(pos, y_best_plot), xytext=(12, 0), textcoords="offset points",
                    va='center', ha='left', fontsize=10, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2b2b2b", alpha=0.9))

    # 4. Divisória Visual
    divisor_x = num_user_algs - 0.5
    ax.axvline(x=divisor_x, color='gray', linestyle=':', linewidth=1.8, alpha=0.8)
    
    # 5. Eixos e Títulos
    todas_labels = nomes_usuario + nomes_lit
    ax.set_xticks(np.arange(len(todas_labels)))
    # Rotação de 45 graus salva espaço horizontal e evita sobreposição de texto!
    ax.set_xticklabels(todas_labels, rotation=40, ha='right', fontsize=11)
    
    ax.set_yscale('log')
    ax.set_ylim(bottom=1e-8)
    ax.set_ylabel('Erro Final (Escala Logarítmica)', fontsize=13, fontweight='bold')
    ax.set_title(f'Comparativo CEC 2014 - F{funcao} (D = {dimensao})', pad=15, fontsize=14, fontweight='bold')
    
    ax.axhline(y=1e-8, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Ótimo de Parada (1e-8)')
    
    # 6. Legenda
    handles, labels = ax.get_legend_handles_labels()
    best_marker = mlines.Line2D([], [], color='gold', marker='*', markeredgecolor='black',
                                linestyle='None', markersize=14, label='Melhor (Best)')
    mean_marker_user = mlines.Line2D([], [], color='white', marker='o', markeredgecolor='black',
                                     linestyle='None', markersize=9, label='Média Modelos')
    mean_marker_lit = mlines.Line2D([], [], color='#2b2b2b', marker='s', markeredgecolor='black',
                                    linestyle='None', markersize=8, label='Média Literatura')
    
    handles.extend([best_marker, mean_marker_user, mean_marker_lit])
    labels.extend(['Melhor (Best)', 'Média Modelos', 'Média Literatura'])
    
    posicao_legenda = 'lower right' if funcao in [2, 4, 6, 9, 13] else 'upper right'
    # Fonte da legenda um pouco menor para não tampar os dados
    ax.legend(handles=handles, labels=labels, loc=posicao_legenda, framealpha=0.95, fontsize=10)
    
    ax.grid(True, which="major", axis="y", linestyle='-', alpha=0.4)
    ax.grid(True, which="minor", axis="y", linestyle=':', alpha=0.2)
    
    plt.tight_layout()
    
    # 7. Salvar
    os.makedirs("graficos", exist_ok=True)
    caminho = f"graficos/Comparativo_Literatura_F{funcao}_D{dimensao}.png"
    plt.savefig(caminho, dpi=300, bbox_inches='tight') # bbox_inches='tight' apara as bordas brancas
    print(f"[OK] Gráfico salvo com sucesso: {caminho}")
    plt.close()

def main():
    pasta_resultados = "results"
    if not os.path.exists(pasta_resultados):
        print(f"Pasta '{pasta_resultados}' não encontrada.")
        return

    arquivos = glob.glob(os.path.join(pasta_resultados, "conv_*_F*_D*.csv"))
    if not arquivos:
        print("Nenhum arquivo de resultados encontrado.")
        return

    dados_agrupados = {}
    padrao = re.compile(r"conv_(.+)_F(\d+)_D(\d+)\.csv")
    
    for arquivo in arquivos:
        match = padrao.search(os.path.basename(arquivo))
        if match:
            algoritmo, funcao, dimensao = match.group(1), int(match.group(2)), int(match.group(3))
            df = pd.read_csv(arquivo, header=None)
            if dimensao not in dados_agrupados: dados_agrupados[dimensao] = {}
            if funcao not in dados_agrupados[dimensao]: dados_agrupados[dimensao][funcao] = {}
            dados_agrupados[dimensao][funcao][algoritmo] = df.iloc[:, -1].values.tolist()

    print("\nGerando Gráficos Comparativos Otimizados para LaTeX...")
    for dim in sorted(dados_agrupados.keys()):
        for func in sorted(dados_agrupados[dim].keys()):
            plotar_comparativo_completo(func, dim, dados_agrupados[dim][func], LITERATURA.get(dim, {}).get(func, {}))
            
if __name__ == "__main__":
    main()