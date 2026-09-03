#ifndef CEC2014_CUH
#define CEC2014_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 1. Classe Host (CPU): Lê os TXTs e joga para a memória da Placa de Vídeo
class GerenciadorDadosCUDA {
public:
    double* d_shifts;
    double* d_matrizes;
    int max_func_id = 13; // Aloca um array de 0 a 13 para mapeamento direto O(1)

    GerenciadorDadosCUDA(int D, const std::vector<int>& funcoesAlvo) {
        // Alocação linear temporária na memória RAM (Host)
        std::vector<double> h_shifts((max_func_id + 1) * D, 0.0);
        std::vector<double> h_matrizes((max_func_id + 1) * D * D, 0.0);

        for (int func : funcoesAlvo) {
            std::string file_shift = "input_data/shift_data_" + std::to_string(func) + ".txt";
            std::ifstream fs(file_shift);
            if (!fs.is_open()) {
                std::cerr << "Erro fatal CUDA: Arquivo " << file_shift << " ausente.\n";
                exit(1);
            }
            for (int i = 0; i < D; ++i) fs >> h_shifts[func * D + i];

            std::string file_mat = "input_data/M_" + std::to_string(func) + "_D" + std::to_string(D) + ".txt";
            std::ifstream fm(file_mat);
            if (!fm.is_open()) {
                std::cerr << "Erro fatal CUDA: Arquivo " << file_mat << " ausente.\n";
                exit(1);
            }
            for (int i = 0; i < D * D; ++i) fm >> h_matrizes[func * D * D + i];
        }

        // Alocação na memória de vídeo (Device - VRAM)
        cudaMalloc(&d_shifts, (max_func_id + 1) * D * sizeof(double));
        cudaMalloc(&d_matrizes, (max_func_id + 1) * D * D * sizeof(double));

        // Transferência via barramento PCIe: RAM -> VRAM
        cudaMemcpy(d_shifts, h_shifts.data(), (max_func_id + 1) * D * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrizes, h_matrizes.data(), (max_func_id + 1) * D * D * sizeof(double), cudaMemcpyHostToDevice);
    }

    ~GerenciadorDadosCUDA() {
        cudaFree(d_shifts);
        cudaFree(d_matrizes);
    }
};

// 2. Função Device (GPU): Matemática pura chamada por milhares de threads simultâneas
__device__ double avaliar_na_gpu(int funcID, const double* x, int D, const double* d_shifts, const double* d_matrizes) {
    // Alocação em memória local/registrador super rápida da GPU (D maximo = 100)
    double z[100];
    double x_shift[100];
    
    // Mapeamento direto (O(1)) do ponteiro da VRAM para a função atual
    const double* shift = &d_shifts[funcID * D];
    const double* M = &d_matrizes[funcID * D * D];

    double escala = 1.0;
    if (funcID == 4) escala = 2.048 / 100.0;
    else if (funcID == 6) escala = 0.5 / 100.0;
    else if (funcID == 7) escala = 600.0 / 100.0;
    else if (funcID == 9) escala = 5.12 / 100.0;
    else if (funcID == 13) escala = 5.0 / 100.0;

    for (int i = 0; i < D; ++i) {
        x_shift[i] = (x[i] - shift[i]) * escala;
        z[i] = 0.0;
    }

    // Aplicação da Matriz de Rotação O(N^2) executada individualmente por cada thread
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            z[i] += M[i * D + j] * x_shift[j];
        }
    }

    double fitness = 0.0;

    switch (funcID) {
        case 2: {
            fitness = z[0] * z[0];
            for (int i = 1; i < D; ++i) fitness += 1e6 * (z[i] * z[i]);
            break;
        }
        case 4: {
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
            double a = 0.5, b = 3.0, sum2 = 0.0;
            int kmax = 20;
            for (int k = 0; k <= kmax; ++k) sum2 += pow(a, k) * cos(2.0 * M_PI * pow(b, k) * 0.5);
            for (int i = 0; i < D; ++i) {
                double sum1 = 0.0;
                for (int k = 0; k <= kmax; ++k) sum1 += pow(a, k) * cos(2.0 * M_PI * pow(b, k) * (z[i] + 0.5));
                fitness += sum1;
            }
            fitness -= D * sum2;
            break;
        }
        case 7: {
            double sum = 0.0, prod = 1.0;
            for (int i = 0; i < D; ++i) {
                sum += (z[i] * z[i]) / 4000.0;
                prod *= cos(z[i] / sqrt(i + 1.0));
            }
            fitness = sum - prod + 1.0;
            break;
        }
        case 9: {
            for (int i = 0; i < D; ++i) fitness += (z[i] * z[i]) - 10.0 * cos(2.0 * M_PI * z[i]) + 10.0;
            break;
        }
        case 13: {
            double sum_sq = 0.0, sum = 0.0;
            for (int i = 0; i < D; ++i) {
                z[i] = z[i] - 1.0;
                sum_sq += z[i] * z[i];
                sum += z[i];
            }
            fitness = pow(fabs(sum_sq - D), 0.25) + (0.5 * sum_sq + sum) / D + 0.5;
            break;
        }
    }
    
    // otimoGlobal é fixo (funcID * 100) na CEC 2014
    return fitness + (funcID * 100.0); 
}

#endif