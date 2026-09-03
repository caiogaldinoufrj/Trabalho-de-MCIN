#ifndef PSO_RL_CUH
#define PSO_RL_CUH

#include <vector>

// Declaração antecipada do estado aleatório da GPU (evita conflitos de header no Host)
struct curandState;

class PSO_RL {
public:
    enum ModoPSO {
        CANONICO = 0,
        ESTATICO_RL = 1,
        TVAC_RL = 2
    };

private:
    int D;
    int tamanhoEnxame;
    int funcID;
    int maxAvaliacoes;
    double limiteErro = 1e-8; // Restrição de precisão do benchmark
    int avaliacoesConsumidas = 0;
    double otimoGlobal;
    ModoPSO modoExecucao;
    unsigned int seedInicial;

    // Coeficientes clássicos
    double w = 0.729;
    double c1 = 1.49445;
    double c2 = 1.49445;

    // Ponteiros para a Memória de Vídeo (VRAM)
    double* d_posicoes;
    double* d_velocidades;
    double* d_pbests;
    double* d_fitnessAtual;
    double* d_pbestFitness;
    
    // Gerenciador de aleatoriedade independente para cada thread na GPU
    curandState* d_estadosCurand;

public:
    PSO_RL(int dimensao, int tamanho, int funcao, ModoPSO modo = TVAC_RL, unsigned int seed = 19937);
    
    // O destrutor é obrigatório em CUDA para executar cudaFree() após cada rodada
    ~PSO_RL();
    
    void inicializar();
    std::vector<double> executar();
};

#endif