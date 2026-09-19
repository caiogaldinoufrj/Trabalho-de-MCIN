import os
import time
import numpy as np
import pandas as pd
from pso_rl import PSO_RL, ModoPSO
from cec2014 import AvaliadorCEC

def calcular_estatisticas(erros):
    # Regra CEC 2014: truncar para zero exato se < 1e-8
    erros_array = np.array(erros)
    erros_array[erros_array < 1e-8] = 0.0
    
    return {
        "Melhor": np.min(erros_array),
        "Pior": np.max(erros_array),
        "Mediana": np.median(erros_array),
        "Media": np.mean(erros_array),
        "DesvioPadrao": np.std(erros_array)
    }

def salvar_historico_csv(nome_arquivo, historico_matriz):
    os.makedirs(os.path.dirname(nome_arquivo), exist_ok=True)
    df = pd.DataFrame(historico_matriz)
    df.to_csv(nome_arquivo, index=False, header=False)

def main():
    funcoes = [2, 4, 6, 7, 9, 13]
    dimensoes = [30, 50]
    num_execucoes = 30

    # Nova estrutura de tuplas: (Modo, Nome, População Inicial)
    modos = [
        (ModoPSO.CANONICO,     "CANONICO",    100),
        (ModoPSO.TVAC_PURO,      "TVAC_PURO",    100),
        (ModoPSO.ESTATICO_RL,  "ESTATICO_RL", 100),
        (ModoPSO.RL_CINEMATICO, "RL_CINEMATICO", 100),
        (ModoPSO.TVAC_RL,      "TVAC_RL",     100),
        (ModoPSO.LPSR_TVAC_RL, "LPSR_RL",     200) 
    ]

    os.makedirs("results", exist_ok=True)
    arquivo_resumo = "results/resumo_estatistico.csv"
    
    # Adicionada a coluna 'PopInicial' no CSV
    with open(arquivo_resumo, "w") as f:
        f.write("Modo,Dimensao,Funcao,PopInicial,Melhor,Pior,Mediana,Media,DesvioPadrao\n")

    for D in dimensoes:
        # Carrega tensores de shift e rotação para a VRAM
        AvaliadorCEC.inicializar_transformador(D, funcoes)

        print(f"\n======================================================")
        print(f" INICIANDO TESTES PARA DIMENSAO D = {D}")
        print(f" Orçamento Máximo de FEs: {10000 * D} avaliações")
        print(f"======================================================")

        for func in funcoes:
            print(f"\n>>> Avaliando Função F{func} (D = {D})")

            # Desempacota as tuplas com o tamanho específico de cada modo
            for modo, nome_modo, tamanho_pop in modos:
                curvas_convergencia = []
                erros_finais = []
                
                tempo_inicio = time.time()

                for run in range(num_execucoes):
                    seed = 19937 + run
                    # Instancia o PSO com a população independente
                    pso = PSO_RL(D, tamanho_pop, func, modo, seed)
                    
                    curva = pso.executar()
                    curvas_convergencia.append(curva)
                    erros_finais.append(curva[-1] if curva else float('inf'))

                tempo_fim = time.time()
                
                # Preenchimento de curvas que convergiram prematuramente (tamanhos diferentes)
                max_len = max(len(c) for c in curvas_convergencia)
                curvas_padronizadas = [c + [c[-1]] * (max_len - len(c)) for c in curvas_convergencia]
                
                arq_conv = f"results/conv_{nome_modo}_F{func}_D{D}.csv"
                salvar_historico_csv(arq_conv, curvas_padronizadas)

                est = calcular_estatisticas(erros_finais)

                with open(arquivo_resumo, "a") as f:
                    f.write(f"{nome_modo},{D},{func},{tamanho_pop},{est['Melhor']:.4e},{est['Pior']:.4e},"
                            f"{est['Mediana']:.4e},{est['Media']:.4e},{est['DesvioPadrao']:.4e}\n")

                # Print atualizado para mostrar a população na tela
                print(f"  [{nome_modo:<11}] Pop: {tamanho_pop:<3} | Média: {est['Media']:.4e} | Mediana: {est['Mediana']:.4e} | "
                      f"Melhor: {est['Melhor']:.4e} | Desvio: {est['DesvioPadrao']:.4e} | "
                      f"Tempo: {tempo_fim - tempo_inicio:.2f}s")

    print("\nTodos os experimentos concluídos com sucesso. Dados em 'results/'.")

if __name__ == "__main__":
    main()