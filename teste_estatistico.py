import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
def testar_estatistica():
    pasta_resultados = "results"
    arquivos = glob.glob(os.path.join(pasta_resultados, "conv_*_F*_D*.csv"))
    
    if not arquivos:
        print(f"Nenhum arquivo encontrado em '{pasta_resultados}'.")
        return

    dados_agrupados = {}
    padrao = re.compile(r"conv_(.+)_F(\d+)_D(\d+)\.csv")
    
    # Ordem de todos os algoritmos na tabela
    ordem_algoritmos = ['CANONICO', 'TVAC_PURO', 'ESTATICO_RL', 'RL_CINEMATICO', 'TVAC_RL', 'LPSR_RL']

    # 1. Carregar os dados
    for arquivo in arquivos:
        match = padrao.search(os.path.basename(arquivo))
        if match:
            alg, func, dim = match.group(1), int(match.group(2)), int(match.group(3))
            df = pd.read_csv(arquivo, header=None)
            erros = df.iloc[:, -1].values
            # Truncamento padrão do CEC
            erros = np.where(erros < 1e-8, 0.0, erros)
            
            if dim not in dados_agrupados: dados_agrupados[dim] = {}
            if func not in dados_agrupados[dim]: dados_agrupados[dim][func] = {}
            dados_agrupados[dim][func][alg] = erros

    print("% Cole este código no seu arquivo LaTeX:")
    print("\\begin{table*}[htbp]")
    print("\\caption{Mediana do Erro Final e Teste de Mann-Whitney U ($\\alpha=0.05$). Símbolos indicam se o modelo de referência (melhor mediana, em negrito) é significativamente superior ($+$) ou estatisticamente equivalente ($\\approx$) ao respectivo algoritmo.}")
    print("\\begin{center}")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{|c|c|ll|ll|ll|ll|ll|ll|}")
    print("\\hline")
    print("\\textbf{D} & \\textbf{F} & \\multicolumn{2}{c|}{\\textbf{Canônico}} & \\multicolumn{2}{c|}{\\textbf{TVAC Puro}} & \\multicolumn{2}{c|}{\\textbf{RL Estático}} & \\multicolumn{2}{c|}{\\textbf{RL Cinemático}} & \\multicolumn{2}{c|}{\\textbf{TVAC-RL}} & \\multicolumn{2}{c|}{\\textbf{LPSR-RL}} \\\\")
    print("\\hline")

    # 2. Processar Estatísticas e Gerar LaTeX
    for dim in sorted(dados_agrupados.keys()):
        for func in sorted(dados_agrupados[dim].keys()):
            dados_func = dados_agrupados[dim][func]
            
            medianas = {}
            for alg in ordem_algoritmos:
                medianas[alg] = np.median(dados_func[alg]) if alg in dados_func else np.inf
                
            melhor_alg = min(medianas, key=medianas.get)
            melhor_mediana = medianas[melhor_alg]
            
            linha_latex = f"{dim} & F{func} "
            
            for alg in ordem_algoritmos:
                if alg not in dados_func:
                    linha_latex += "& - & "
                    continue
                
                erros_alg = dados_func[alg]
                mediana_atual = medianas[alg]
                
                # Formatação científica
                if mediana_atual == 0.0:
                    str_mediana = "0.00e+00"
                else:
                    str_mediana = f"{mediana_atual:.2e}"
                
                # Negrito para o melhor
                if alg == melhor_alg:
                    str_mediana = f"\\textbf{{{str_mediana}}}"
                    sinal = " " # Referência
                else:
                    # Teste de Wilcoxon para amostras pareadas
                    erros_melhor = dados_func[melhor_alg]
                    
                    # Diferença entre os arrays
                    diferencas = erros_melhor - erros_alg
                    
                    # Se não houver diferença nenhuma (ou for numericamente 0 em todos os casos), é empate
                    if np.allclose(diferencas, 0):
                        sinal = "$\\approx$"
                    else:
                        try:
                            # alternative='less' testa se erros_melhor é estatisticamente menor que erros_alg
                            stat, p_value = wilcoxon(erros_melhor, erros_alg, alternative='less')
                            if p_value < 0.05:
                                sinal = "$+$"  # Modelo de referência é significativamente melhor
                            else:
                                sinal = "$\\approx$"  # Empate estatístico
                        except ValueError:
                            # O Wilcoxon pode lançar erro se todas as diferenças não nulas forem perfeitamente simétricas 
                            # ou muito pequenas. Nesses casos raros, consideramos empate.
                            sinal = "$\\approx$"
                            
                linha_latex += f"& {str_mediana} & {sinal} "
            
            linha_latex += "\\\\"
            print(linha_latex)
        print("\\hline")
        
    print("\\end{tabular}}")
    print("\\label{tab:estatistica}")
    print("\\end{center}")
    print("\\end{table*}")

if __name__ == "__main__":
    testar_estatistica()