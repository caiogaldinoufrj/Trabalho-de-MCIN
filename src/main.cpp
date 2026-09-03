#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <limits>
#include "PSO_RL.h"
#include "CEC2014.h" // Importação obrigatória para carregar as matrizes

struct EstatisticasRun {
    double melhor;
    double pior;
    double mediana;
    double media;
    double desvioPadrao;
};

EstatisticasRun calcularEstatisticas(std::vector<double> erros) {
    EstatisticasRun est;
    if (erros.empty()) return est;

    // Regra CEC 2014: erro inferior a 1e-8 é truncado para zero exato[cite: 1]
    for (double& e : erros) {
        if (e < 1e-8) e = 0.0;
    }

    std::sort(erros.begin(), erros.end());

    est.melhor = erros.front();
    est.pior = erros.back();
    
    size_t n = erros.size();
    if (n % 2 == 0) {
        est.mediana = (erros[n / 2 - 1] + erros[n / 2]) / 2.0;
    } else {
        est.mediana = erros[n / 2];
    }

    double soma = std::accumulate(erros.begin(), erros.end(), 0.0);
    est.media = soma / n;

    double somaQuadrados = 0.0;
    for (double e : erros) {
        somaQuadrados += (e - est.media) * (e - est.media);
    }
    est.desvioPadrao = std::sqrt(somaQuadrados / n);

    return est;
}

void salvarHistoricoCSV(const std::string& nomeArquivo, const std::vector<std::vector<double>>& historicoMatriz) {
    std::ofstream arquivo(nomeArquivo);
    if (!arquivo.is_open()) {
        std::cerr << "Erro ao criar: " << nomeArquivo << "\n";
        return;
    }
    for (const auto& runHistorico : historicoMatriz) {
        for (size_t i = 0; i < runHistorico.size(); ++i) {
            arquivo << runHistorico[i] << (i < runHistorico.size() - 1 ? "," : "");
        }
        arquivo << "\n";
    }
    arquivo.close();
}

int main() {
    const std::vector<int> funcoes = {2, 4, 6, 7, 9, 13};
    const std::vector<int> dimensoes = {30, 50};
    const int numExecucoes = 30; 
    const int tamanhoEnxame = 100;

    const std::vector<std::pair<PSO_RL::ModoPSO, std::string>> modos = {
        {PSO_RL::CANONICO, "CANONICO"},
        {PSO_RL::ESTATICO_RL, "ESTATICO_RL"},
        {PSO_RL::TVAC_RL, "TVAC_RL"}
    };

    std::ofstream arqResumo("results/resumo_estatistico.csv");
    arqResumo << "Modo,Dimensao,Funcao,Melhor,Pior,Mediana,Media,DesvioPadrao\n";

    std::cout << std::scientific << std::setprecision(4);

    for (int D : dimensoes) {
        // Carrega as matrizes de rotação e vetores shift para a dimensão atual antes de rodar os enxames
        AvaliadorCEC::inicializarTransformador(D, funcoes);

        std::cout << "\n======================================================\n";
        std::cout << " INICIANDO TESTES PARA DIMENSAO D = " << D << "\n";
        std::cout << " Orçamento Máximo de FEs: " << 10000 * D << " avaliações\n";
        std::cout << "======================================================\n";

        for (int func : funcoes) {
            std::cout << "\n>>> Avaliando Função F" << func << " (D = " << D << ")\n";

            for (const auto& [modo, nomeModo] : modos) {
                std::vector<std::vector<double>> curvasConvergencia;
                std::vector<double> errosFinais;
                curvasConvergencia.reserve(numExecucoes);
                errosFinais.reserve(numExecucoes);

                for (int run = 0; run < numExecucoes; ++run) {
                    unsigned int seed = 19937 + run;

                    PSO_RL pso(D, tamanhoEnxame, func, modo, seed);
                    std::vector<double> curva = pso.executar();
                    
                    curvasConvergencia.push_back(curva);

                    double erroFinal = curva.empty() ? std::numeric_limits<double>::max() : curva.back();
                    errosFinais.push_back(erroFinal);
                }

                std::string arqConv = "results/conv_" + nomeModo + "_F" + std::to_string(func) + "_D" + std::to_string(D) + ".csv";
                salvarHistoricoCSV(arqConv, curvasConvergencia);

                EstatisticasRun est = calcularEstatisticas(errosFinais);

                arqResumo << nomeModo << ","
                          << D << ","
                          << func << ","
                          << est.melhor << ","
                          << est.pior << ","
                          << est.mediana << ","
                          << est.media << ","
                          << est.desvioPadrao << "\n";
                arqResumo.flush();

                std::cout << "  [" << std::setw(11) << std::left << nomeModo << "] "
                          << "Média: " << std::setw(11) << est.media << " | "
                          << "Mediana: " << std::setw(11) << est.mediana << " | "
                          << "Melhor: " << std::setw(11) << est.melhor << " | "
                          << "Desvio: " << est.desvioPadrao << "\n";
            }
        }
    }

    arqResumo.close();
    std::cout << "\nTodos os experimentos foram concluídos com sucesso. Dados em 'results/'.\n";
    return 0;
}