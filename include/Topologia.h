#ifndef TOPOLOGIA_H
#define TOPOLOGIA_H

#include <vector>
#include <memory>
#include <random>
#include <limits>

struct Particula {
    std::vector<double> posicao;
    std::vector<double> velocidade;
    std::vector<double> pbest;
    double fitnessAtual;
    double pbestFitness;
    
    Particula(int D) : 
        posicao(D, 0.0), velocidade(D, 0.0), pbest(D, 0.0), 
        fitnessAtual(0.0), pbestFitness(std::numeric_limits<double>::max()) {}
};

class Topologia {
public:
    virtual ~Topologia() = default;
    virtual std::vector<double> obterLider(int indice, const std::vector<std::unique_ptr<Particula>>& enxame, std::mt19937& gen) = 0;
};

class TopologiaEstrela : public Topologia {
public:
    // Omitindo os nomes 'indice' e 'gen' para evitar avisos do compilador
    std::vector<double> obterLider(int /*indice*/, const std::vector<std::unique_ptr<Particula>>& enxame, std::mt19937& /*gen*/) override {
        int idxMelhor = 0;
        double melhorFit = enxame[0]->pbestFitness;
        for (size_t i = 1; i < enxame.size(); ++i) {
            if (enxame[i]->pbestFitness < melhorFit) {
                melhorFit = enxame[i]->pbestFitness;
                idxMelhor = i;
            }
        }
        return enxame[idxMelhor]->pbest;
    }
};

class TopologiaAnel : public Topologia {
public:
    // Omitindo o nome 'gen'
    std::vector<double> obterLider(int indice, const std::vector<std::unique_ptr<Particula>>& enxame, std::mt19937& /*gen*/) override {
        int n = enxame.size();
        int vizinhoEsq = (indice - 1 + n) % n;
        int vizinhoDir = (indice + 1) % n;
        
        int idxLider = indice;
        if (enxame[vizinhoEsq]->pbestFitness < enxame[idxLider]->pbestFitness) idxLider = vizinhoEsq;
        if (enxame[vizinhoDir]->pbestFitness < enxame[idxLider]->pbestFitness) idxLider = vizinhoDir;
        
        return enxame[idxLider]->pbest;
    }
};

class TopologiaAleatoria : public Topologia {
public:
    std::vector<double> obterLider(int indice, const std::vector<std::unique_ptr<Particula>>& enxame, std::mt19937& gen) override {
        std::uniform_int_distribution<int> dis(0, enxame.size() - 1);
        int aleatorio = dis(gen);
        if (enxame[aleatorio]->pbestFitness < enxame[indice]->pbestFitness) {
            return enxame[aleatorio]->pbest;
        }
        return enxame[indice]->pbest;
    }
};

#endif