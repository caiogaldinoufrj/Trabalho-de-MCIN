import numpy as np

try:
    import cupy as cp
except ImportError:
    print("Aviso: CuPy não detectado. Operando na CPU com NumPy.")
    import numpy as cp

from agente_rl import AgenteRL
from cec2014 import AvaliadorCEC

class ModoPSO:
    CANONICO = 0
    ESTATICO_RL = 1
    TVAC_RL = 2

class PSO_RL:
    def __init__(self, dimensao: int, tamanho: int, funcao: int, modo: int = ModoPSO.TVAC_RL, seed: int = 19937):
        self.D = dimensao
        self.tamanho_enxame = tamanho
        self.func_id = funcao
        self.modo_execucao = modo
        
        # Gerador de números pseudoaleatórios na VRAM da GPU
        self.rng = cp.random.default_rng(seed)
        
        self.max_avaliacoes = 10000 * self.D
        self.limite_erro = 1e-8
        self.otimo_global = self.func_id * 100.0
        self.avaliacoes_consumidas = 0
        
        # Coeficientes estáticos
        self.w = 0.729
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.Vmax = 20.0
        
        # Matrizes do Enxame (alocadas na inicialização)
        self.posicoes = None
        self.velocidades = None
        self.pbests = None
        self.pbest_fitness = None

    def inicializar(self):
        # Alocação matricial (População x Dimensão) rodando 100% na GPU
        self.posicoes = self.rng.uniform(-100.0, 100.0, (self.tamanho_enxame, self.D))
        self.velocidades = cp.zeros((self.tamanho_enxame, self.D))
        self.pbests = cp.copy(self.posicoes)
        
        self.pbest_fitness = AvaliadorCEC.avaliar(self.func_id, self.posicoes)
        self.avaliacoes_consumidas = self.tamanho_enxame

    def _obter_lideres(self, acao: int) -> cp.ndarray:
        if acao == 0:  # Topologia Estrela (Gbest)
            idx_melhor = cp.argmin(self.pbest_fitness)
            return self.pbests[idx_melhor] # Array (D,) faz broadcasting para (População, D) automaticamente
            
        elif acao == 1:  # Topologia Anel (Lbest)
            fit_esq = cp.roll(self.pbest_fitness, 1)
            fit_dir = cp.roll(self.pbest_fitness, -1)
            
            matriz_fits = cp.vstack((self.pbest_fitness, fit_esq, fit_dir))
            melhor_relativo = cp.argmin(matriz_fits, axis=0)
            
            indices = cp.arange(self.tamanho_enxame)
            indices_esq = (indices - 1) % self.tamanho_enxame
            indices_dir = (indices + 1) % self.tamanho_enxame
            
            indices_lideres = cp.choose(melhor_relativo, [indices, indices_esq, indices_dir])
            return self.pbests[indices_lideres]
            
        elif acao == 2:  # Topologia Aleatória
            indices = cp.arange(self.tamanho_enxame)
            indices_rand = self.rng.integers(0, self.tamanho_enxame, self.tamanho_enxame)
            
            venceu_rand = self.pbest_fitness[indices_rand] < self.pbest_fitness
            indices_lideres = cp.where(venceu_rand, indices_rand, indices)
            return self.pbests[indices_lideres]

    def executar(self) -> list:
        self.inicializar()
        
        # O Agente RL (CPU) recebe uma semente derivada para quebrar simetria entre as 30 runs independentes
        agente = AgenteRL(seed=int(self.rng.integers(0, 2**31)))
        
        melhor_erro_atual = float(cp.min(self.pbest_fitness)) - self.otimo_global
        erro_anterior = melhor_erro_atual
        historico_erro = []
        
        while self.avaliacoes_consumidas < self.max_avaliacoes and melhor_erro_atual >= self.limite_erro:
            estado_atual = 0
            acao_escolhida = 0 # Default: Estrela
            
            if self.modo_execucao != ModoPSO.CANONICO:
                estado_atual = agente.discretizar_estado(erro_anterior, melhor_erro_atual)
                acao_escolhida = agente.selecionar_acao(estado_atual)
            
            w_atual, c1_atual, c2_atual = self.w, self.c1, self.c2
            if self.modo_execucao == ModoPSO.TVAC_RL:
                progresso = self.avaliacoes_consumidas / self.max_avaliacoes
                w_atual  = 0.9 - (0.9 - 0.4) * progresso
                c1_atual = 2.5 - (2.5 - 0.5) * progresso
                c2_atual = 0.5 + (2.5 - 0.5) * progresso
            
            lideres = self._obter_lideres(acao_escolhida)
            
            r1 = self.rng.random((self.tamanho_enxame, self.D))
            r2 = self.rng.random((self.tamanho_enxame, self.D))
            
            # Equação de movimento calculada integralmente de forma paralela na GPU
            self.velocidades = (w_atual * self.velocidades + 
                                c1_atual * r1 * (self.pbests - self.posicoes) + 
                                c2_atual * r2 * (lideres - self.posicoes))
            
            self.velocidades = cp.clip(self.velocidades, -self.Vmax, self.Vmax)
            self.posicoes += self.velocidades
            
            # Tratamento de fronteiras
            ultrapassou_max = self.posicoes > 100.0
            ultrapassou_min = self.posicoes < -100.0
            self.posicoes = cp.clip(self.posicoes, -100.0, 100.0)
            
            rebote = ultrapassou_max | ultrapassou_min
            self.velocidades = cp.where(rebote, self.velocidades * -0.5, self.velocidades)
            
            if self.avaliacoes_consumidas < self.max_avaliacoes:
                fitness_atual = AvaliadorCEC.avaliar(self.func_id, self.posicoes)
                self.avaliacoes_consumidas += self.tamanho_enxame
                
                # Substituição vetorizada do pbest
                melhorou = fitness_atual < self.pbest_fitness
                self.pbest_fitness = cp.where(melhorou, fitness_atual, self.pbest_fitness)
                # Expandindo a máscara 'melhorou' para copiar posições ao invés de usar loops `for d em D`
                self.pbests = cp.where(melhorou[:, cp.newaxis], self.posicoes, self.pbests)
            
            erro_anterior = melhor_erro_atual
            melhor_erro_atual = float(cp.min(self.pbest_fitness)) - self.otimo_global
            historico_erro.append(melhor_erro_atual)
            
            if self.modo_execucao != ModoPSO.CANONICO:
                proximo_estado = agente.discretizar_estado(erro_anterior, melhor_erro_atual)
                recompensa = agente.calcular_recompensa(estado_atual, proximo_estado, melhor_erro_atual)
                agente.atualizar_q(estado_atual, acao_escolhida, recompensa, proximo_estado)
                
        return historico_erro