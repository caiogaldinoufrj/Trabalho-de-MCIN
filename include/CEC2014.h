#ifndef CEC2014_H
#define CEC2014_H

#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <memory>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class TransformadorCEC {
private:
    std::map<int, std::vector<double>> shifts;
    std::map<int, std::vector<std::vector<double>>> matrizes;
    int D;

    void carregarShift(int funcID) {
        std::string filename = "input_data/shift_data_" + std::to_string(funcID) + ".txt";
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Erro fatal: Arquivo de shift " << filename << " ausente na raiz.\n";
            exit(1);
        }
        std::vector<double> O(D);
        for (int i = 0; i < D; ++i) file >> O[i];
        shifts[funcID] = std::move(O);
    }

    void carregarMatriz(int funcID) {
        std::string filename = "input_data/M_" + std::to_string(funcID) + "_D" + std::to_string(D) + ".txt";
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Erro fatal: Arquivo de rotacao " << filename << " ausente na raiz.\n";
            exit(1);
        }
        std::vector<std::vector<double>> M(D, std::vector<double>(D));
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) file >> M[i][j];
        }
        matrizes[funcID] = std::move(M);
    }

public:
    TransformadorCEC(int dimensao, const std::vector<int>& funcoesAlvo) : D(dimensao) {
        for (int func : funcoesAlvo) {
            carregarShift(func);
            carregarMatriz(func); // Correção: A F2 do CEC 2014 original roda matriz sim
        }
    }

    // Aplica z = M_i * ((x - o_i) * escala)
    std::vector<double> transformar(int funcID, const std::vector<double>& x, double escala) {
        std::vector<double> z(D, 0.0);
        std::vector<double> x_shift(D, 0.0);
        
        const auto& shift = shifts[funcID];
        const auto& M = matrizes[funcID];

        for (int i = 0; i < D; ++i) {
            x_shift[i] = (x[i] - shift[i]) * escala;
        }

        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                z[i] += M[i][j] * x_shift[j];
            }
        }
        return z;
    }
};

class AvaliadorCEC {
private:
    inline static std::unique_ptr<TransformadorCEC> transformador = nullptr;
    
public:
    static void inicializarTransformador(int D, const std::vector<int>& funcoesAlvo) {
        transformador = std::make_unique<TransformadorCEC>(D, funcoesAlvo);
    }

    static double avaliar(int funcID, const std::vector<double>& x, double otimoGlobal) {
        double fitness = 0.0;
        int D = x.size();
        std::vector<double> z;

        switch (funcID) {
            case 2: {
                z = transformador->transformar(funcID, x, 1.0);
                fitness = z[0] * z[0];
                for (int i = 1; i < D; ++i) fitness += 1e6 * (z[i] * z[i]);
                break;
            }
            case 4: {
                z = transformador->transformar(funcID, x, 2.048 / 100.0);
                for (int i = 0; i < D - 1; ++i) {
                    double z_i = z[i] + 1.0;
                    double z_next = z[i+1] + 1.0;
                    double term1 = (z_i * z_i) - z_next;
                    double term2 = z_i - 1.0;
                    fitness += 100.0 * (term1 * term1) + (term2 * term2);
                }
                break;
            }
            case 6: {
                z = transformador->transformar(funcID, x, 0.5 / 100.0);
                double a = 0.5, b = 3.0, sum2 = 0.0;
                int kmax = 20;
                for (int k = 0; k <= kmax; ++k) sum2 += std::pow(a, k) * std::cos(2.0 * M_PI * std::pow(b, k) * 0.5);
                for (int i = 0; i < D; ++i) {
                    double sum1 = 0.0;
                    for (int k = 0; k <= kmax; ++k) sum1 += std::pow(a, k) * std::cos(2.0 * M_PI * std::pow(b, k) * (z[i] + 0.5));
                    fitness += sum1;
                }
                fitness -= D * sum2;
                break;
            }
            case 7: {
                z = transformador->transformar(funcID, x, 600.0 / 100.0);
                double sum = 0.0, prod = 1.0;
                for (int i = 0; i < D; ++i) {
                    sum += (z[i] * z[i]) / 4000.0;
                    prod *= std::cos(z[i] / std::sqrt(i + 1.0));
                }
                fitness = sum - prod + 1.0;
                break;
            }
            case 9: {
                z = transformador->transformar(funcID, x, 5.12 / 100.0);
                for (int i = 0; i < D; ++i) fitness += (z[i] * z[i]) - 10.0 * std::cos(2.0 * M_PI * z[i]) + 10.0;
                break;
            }
            case 13: {
                z = transformador->transformar(funcID, x, 5.0 / 100.0);
                double sum_sq = 0.0, sum = 0.0;
                for (int i = 0; i < D; ++i) {
                    z[i] = z[i] - 1.0; // Correção: Deslocamento forçado na linha 477 do C original
                    sum_sq += z[i] * z[i];
                    sum += z[i];
                }
                fitness = std::pow(std::abs(sum_sq - D), 0.25) + (0.5 * sum_sq + sum) / D + 0.5;
                break;
            }
            default: {
                std::cerr << "Erro: Funcao " << funcID << " nao implementada no AvaliadorCEC.\n";
                exit(1);
            }
        }
        return fitness + otimoGlobal;
    }
};

#endif