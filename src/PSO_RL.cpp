#include "PSO_RL.h"
#include "AgenteRL.h"
#include "CEC2014.h"
#include <algorithm>
#include <cmath>

PSO_RL::PSO_RL(int dimensao, int tamanho, int funcao, ModoPSO modo, unsigned int seed) 
    : D(dimensao), tamanhoEnxame(tamanho), funcID(funcao), modoExecucao(modo) {
    maxAvaliacoes = 10000 * D; // Limite rigoroso do benchmark[cite: 1]
    otimoGlobal = funcID * 100.0; // Deslocamento característico da suíte[cite: 1]
    gen = std::mt19937(seed); // Inicialização estritamente determinística
}

void PSO_RL::inicializar() {
    std::uniform_real_distribution<double> dis(-100.0, 100.0); // Espaço de busca simétrico[cite: 1]
    enxame.clear();
    avaliacoesConsumidas = 0;

    for (int i = 0; i < tamanhoEnxame; ++i) {
        auto p = std::make_unique<Particula>(D);
        for (int j = 0; j < D; ++j) {
            p->posicao[j] = dis(gen);
            p->pbest[j] = p->posicao[j];
            p->velocidade[j] = 0.0;
        }
        p->fitnessAtual = AvaliadorCEC::avaliar(funcID, p->posicao, otimoGlobal);
        p->pbestFitness = p->fitnessAtual;
        avaliacoesConsumidas++;
        enxame.push_back(std::move(p));
    }
}

std::vector<double> PSO_RL::executar() {
    inicializar();
    
    // O Agente herda um salto do Mersenne Twister do PSO, garantindo variação controlada
    AgenteRL agente(gen());
    
    std::vector<std::unique_ptr<Topologia>> topologias;
    topologias.push_back(std::make_unique<TopologiaEstrela>());   // Ação 0
    topologias.push_back(std::make_unique<TopologiaAnel>());      // Ação 1
    topologias.push_back(std::make_unique<TopologiaAleatoria>()); // Ação 2

    double melhorErroAtual = std::numeric_limits<double>::max();
    for (const auto& p : enxame) {
        double erroP = std::abs(p->pbestFitness - otimoGlobal);
        if (erroP < melhorErroAtual) melhorErroAtual = erroP;
    }

    double erroAnterior = melhorErroAtual;
    std::vector<double> historicoErro;
    std::uniform_real_distribution<double> rand_dist(0.0, 1.0);

    const double Vmax = 20.0; // 20% do semieixo para impedir divergência

    // Critérios de parada: esgotamento de avaliações ou erro < 10^-8[cite: 1]
    while (avaliacoesConsumidas < maxAvaliacoes && melhorErroAtual >= limiteErro) {
        
        int estadoAtual = 0;
        int acaoEscolhida = 0; // Padrão: Estrela (Gbest)

        if (modoExecucao != CANONICO) {
            estadoAtual = agente.discretizarEstado(erroAnterior, melhorErroAtual);
            acaoEscolhida = agente.selecionarAcao(estadoAtual);
        }

        double w_atual = this->w;
        double c1_atual = this->c1;
        double c2_atual = this->c2;

        if (modoExecucao == TVAC_RL) {
            double progresso = static_cast<double>(avaliacoesConsumidas) / maxAvaliacoes;
            w_atual  = 0.9 - (0.9 - 0.4) * progresso;  
            c1_atual = 2.5 - (2.5 - 0.5) * progresso;  
            c2_atual = 0.5 + (2.5 - 0.5) * progresso;  
        }

        for (size_t i = 0; i < enxame.size(); ++i) {
            std::vector<double> lider = topologias[acaoEscolhida]->obterLider(i, enxame, gen);
            
            for (int d = 0; d < D; ++d) {
                double r1 = rand_dist(gen);
                double r2 = rand_dist(gen);

                enxame[i]->velocidade[d] = w_atual * enxame[i]->velocidade[d] + 
                                           c1_atual * r1 * (enxame[i]->pbest[d] - enxame[i]->posicao[d]) + 
                                           c2_atual * r2 * (lider[d] - enxame[i]->posicao[d]);
                
                if (enxame[i]->velocidade[d] > Vmax)  enxame[i]->velocidade[d] = Vmax;
                if (enxame[i]->velocidade[d] < -Vmax) enxame[i]->velocidade[d] = -Vmax;

                enxame[i]->posicao[d] += enxame[i]->velocidade[d];
                
                // Fronteiras do CEC 2014 com amortecimento de rebote[cite: 1]
                if (enxame[i]->posicao[d] > 100.0) {
                    enxame[i]->posicao[d] = 100.0;
                    enxame[i]->velocidade[d] = -enxame[i]->velocidade[d] * 0.5;
                }
                if (enxame[i]->posicao[d] < -100.0) {
                    enxame[i]->posicao[d] = -100.0;
                    enxame[i]->velocidade[d] = -enxame[i]->velocidade[d] * 0.5;
                }
            }
            
            if (avaliacoesConsumidas < maxAvaliacoes) {
                enxame[i]->fitnessAtual = AvaliadorCEC::avaliar(funcID, enxame[i]->posicao, otimoGlobal);
                avaliacoesConsumidas++;
                
                if (enxame[i]->fitnessAtual < enxame[i]->pbestFitness) {
                    enxame[i]->pbestFitness = enxame[i]->fitnessAtual;
                    enxame[i]->pbest = enxame[i]->posicao;
                }
            }
        }

        erroAnterior = melhorErroAtual;
        for (const auto& p : enxame) {
            double erroP = std::abs(p->pbestFitness - otimoGlobal);
            if (erroP < melhorErroAtual) melhorErroAtual = erroP;
        }
        
        historicoErro.push_back(melhorErroAtual);
        
        if (modoExecucao != CANONICO) {
            int proximoEstado = agente.discretizarEstado(erroAnterior, melhorErroAtual);
            double recompensa = agente.calcularRecompensa(estadoAtual, proximoEstado, melhorErroAtual);
            agente.atualizarQ(estadoAtual, acaoEscolhida, recompensa, proximoEstado);
        }
    }
    
    return historicoErro;
}