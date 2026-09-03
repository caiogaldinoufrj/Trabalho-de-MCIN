#ifndef AGENTE_RL_H
#define AGENTE_RL_H

#include <vector>
#include <random>
#include <algorithm>

class AgenteRL {
private:
    std::vector<std::vector<double>> q_table;
    double alpha = 0.1; 
    double gamma = 0.9; 
    double epsilon = 0.2; 
    int numEstados = 4;
    int numAcoes = 3;
    std::mt19937 gen;

public:
    // Recebe a semente herdada da rodada atual para garantir reproducibilidade estrita
    explicit AgenteRL(unsigned int seed = 19937) {
        gen = std::mt19937(seed);
        q_table.resize(numEstados, std::vector<double>(numAcoes));
        
        std::uniform_real_distribution<double> dist_q(-0.1, 0.1);
        for (int i = 0; i < numEstados; ++i) {
            for (int j = 0; j < numAcoes; ++j) {
                q_table[i][j] = dist_q(gen);
            }
        }
    }

    int discretizarEstado(double erroAnterior, double erroAtual) {
        if (erroAnterior <= 1e-12) return 3; // Blindagem contra subnormais e zero
        
        double delta = (erroAnterior - erroAtual) / erroAnterior;
        if (delta > 0.05) return 0; // Decaimento acentuado
        if (delta > 0.01) return 1; // Decaimento moderado
        if (delta > 0.0)  return 2; // Decaimento suave
        return 3;                   // Estagnação/platô
    }

    int selecionarAcao(int estadoAtual) {
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        if (dis(gen) < epsilon) {
            std::uniform_int_distribution<int> acaoDis(0, numAcoes - 1);
            return acaoDis(gen);
        }
        auto it = std::max_element(q_table[estadoAtual].begin(), q_table[estadoAtual].end());
        return std::distance(q_table[estadoAtual].begin(), it);
    }

    void atualizarQ(int estadoAtual, int acao, double recompensa, int proximoEstado) {
        double maxQProximo = *std::max_element(q_table[proximoEstado].begin(), q_table[proximoEstado].end());
        q_table[estadoAtual][acao] += alpha * (recompensa + gamma * maxQProximo - q_table[estadoAtual][acao]);
    }

    double calcularRecompensa(int estadoAtual, int proximoEstado, double erro) {
        if (erro < 1e-8) return 100.0; // Critério de corte atingido
        if (estadoAtual == 3 && (proximoEstado == 0 || proximoEstado == 1)) return 10.0; // Recompensa por romper inércia
        if (proximoEstado == 3) return -5.0; // Penalidade por estagnação
        return -1.0;                         // Custo temporal por transição
    }
};

#endif