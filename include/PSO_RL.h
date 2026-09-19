#ifndef PSO_RL_H
#define PSO_RL_H

#include <vector>
#include <memory>
#include <random>
#include "Topologia.h"

class PSO_RL {
public:
    enum ModoPSO {
        CANONICO = 0,   // Parâmetros estáticos, topologia Estrela fixa
        ESTATICO_RL = 1,// Parâmetros estáticos, topologia controlada pelo Q-Learning
        TVAC_RL = 2     // Parâmetros dinâmicos (TVAC), topologia controlada pelo Q-Learning
    };

private:
    int D;
    int tamanhoEnxame;
    int funcID;
    int maxAvaliacoes;
    double limiteErro = 1e-8; // Restrição de precisão do trabalho[cite: 2]
    int avaliacoesConsumidas = 0;
    double otimoGlobal;
    ModoPSO modoExecucao;
    std::mt19937 gen;
    std::vector<std::unique_ptr<Particula>> enxame;
    
    // Coeficientes para os modos estáticos (CANONICO e ESTATICO_RL)
    double w = 0.729;
    double c1 = 1.49445;
    double c2 = 1.49445;

public:
    // Construtor com TVAC_RL como padrão caso nenhum modo seja informado
    PSO_RL(int dimensao, int tamanho, int funcao, ModoPSO modo = TVAC_RL, unsigned int seed = 19937);
    
    void inicializar();
    std::vector<double> executar();
};

#endif