import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configurações do experimento
funcoes = [2, 4, 6, 7, 9, 13]
dimensoes = [30, 50]
modos = ["CANONICO", "ESTATICO_RL", "TVAC_RL"]
cores = {"CANONICO": "red", "ESTATICO_RL": "blue", "TVAC_RL": "green"}

os.makedirs("results/plots", exist_ok=True)

def carregar_e_padronizar(filepath):
    """Lê CSV com linhas de tamanhos diferentes (parada antecipada) e preenche com o erro final."""
    linhas = []
    max_len = 0
    with open(filepath, 'r') as f:
        for line in f:
            valores = [float(x) for x in line.strip().split(',') if x]
            if valores:
                linhas.append(valores)
                if len(valores) > max_len:
                    max_len = len(valores)
    
    # Preenche as execuções que convergiram cedo com o último valor
    matriz = np.zeros((len(linhas), max_len))
    for i, linha in enumerate(linhas):
        matriz[i, :len(linha)] = linha
        matriz[i, len(linha):] = linha[-1]
    
    return np.mean(matriz, axis=0) # Retorna a curva média das 30 execuções

for D in dimensoes:
    for func in funcoes:
        plt.figure(figsize=(10, 6))
        plt.title(f"Convergência - Função F{func} (D={D})", fontsize=14)
        plt.xlabel("Gerações", fontsize=12)
        plt.ylabel("Erro (Log Scale)", fontsize=12)
        
        dados_encontrados = False
        
        for modo in modos:
            filepath = f"results/conv_{modo}_F{func}_D{D}.csv"
            if os.path.exists(filepath):
                curva_media = carregar_e_padronizar(filepath)
                # Adiciona offset mínimo para evitar erro no log10(0) caso atinja 0.0 exato
                plt.plot(np.maximum(curva_media, 1e-8), label=modo, color=cores[modo], linewidth=2)
                dados_encontrados = True
                
        if dados_encontrados:
            plt.yscale('log')
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend(fontsize=11)
            plt.tight_layout()
            
            output_path = f"results/plots/Convergencia_F{func}_D{D}.png"
            plt.savefig(output_path, dpi=300)
            print(f"Gráfico salvo: {output_path}")
        plt.close()

print("\nGeração de gráficos concluída. Verifique a pasta 'results/plots/'.")