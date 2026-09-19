import numpy as np
from scipy.stats import qmc

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
    LPSR_TVAC_RL = 3
    RL_CINEMATICO = 4
    TVAC_PURO = 5 
    

class PSO_RL:
    def __init__(self, dimensao: int, tamanho: int, funcao: int, modo: int = ModoPSO.TVAC_RL, seed: int = 19937):
        self.D = dimensao
        self.tamanho_inicial = tamanho      # O tamanho exato passado lá no main.py para este algoritmo
        self.tamanho_minimo = 10            # Piso de corte para o modo LPSR
        self.tamanho_enxame = tamanho
        self.func_id = funcao
        self.modo_execucao = modo
        
        self.rng = cp.random.default_rng(seed)
        
        self.max_avaliacoes = 10000 * self.D
        self.limite_erro = 1e-8
        self.otimo_global = self.func_id * 100.0
        self.avaliacoes_consumidas = 0
        
        self.w = 0.729
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.Vmax = 20.0
        
        self.posicoes = None
        self.velocidades = None
        self.pbests = None
        self.pbest_fitness = None

    def inicializar(self):
        int_seed = int(self.rng.integers(0, 2**31))
        sampler = qmc.LatinHypercube(d=self.D, seed=int_seed)
        amostras_cpu = sampler.random(n=self.tamanho_enxame) 
        
        self.posicoes = cp.asarray(-100.0 + 200.0 * amostras_cpu)
        self.velocidades = cp.zeros((self.tamanho_enxame, self.D))
        self.pbests = cp.copy(self.posicoes)
        
        self.pbest_fitness = AvaliadorCEC.avaliar(self.func_id, self.posicoes)
        self.avaliacoes_consumidas = self.tamanho_enxame

    def _obter_lideres(self, acao: int) -> cp.ndarray:
        if acao == 0:  
            idx_melhor = cp.argmin(self.pbest_fitness)
            return self.pbests[idx_melhor] 
        elif acao == 1:  
            fit_esq = cp.roll(self.pbest_fitness, 1)
            fit_dir = cp.roll(self.pbest_fitness, -1)
            matriz_fits = cp.vstack((self.pbest_fitness, fit_esq, fit_dir))
            melhor_relativo = cp.argmin(matriz_fits, axis=0)
            indices = cp.arange(self.tamanho_enxame)
            indices_esq = (indices - 1) % self.tamanho_enxame
            indices_dir = (indices + 1) % self.tamanho_enxame
            indices_lideres = cp.choose(melhor_relativo, [indices, indices_esq, indices_dir])
            return self.pbests[indices_lideres]
        elif acao == 2:  
            indices = cp.arange(self.tamanho_enxame)
            indices_rand = self.rng.integers(0, self.tamanho_enxame, self.tamanho_enxame)
            venceu_rand = self.pbest_fitness[indices_rand] < self.pbest_fitness
            indices_lideres = cp.where(venceu_rand, indices_rand, indices)
            return self.pbests[indices_lideres]

    def executar(self) -> list:
        self.inicializar()
        
        agente = AgenteRL(seed=int(self.rng.integers(0, 2**31)))
        melhor_erro_atual = float(cp.min(self.pbest_fitness)) - self.otimo_global
        erro_anterior = melhor_erro_atual
        historico_erro = []
        
        div_salva = 0.0
        
        while self.avaliacoes_consumidas < self.max_avaliacoes and melhor_erro_atual >= self.limite_erro:
            progresso = self.avaliacoes_consumidas / self.max_avaliacoes
            
            # =========================================================
            # LPSR - Redução Linear de População (EXCLUSIVO PARA MODO 3)
            # =========================================================
            if self.modo_execucao == ModoPSO.LPSR_TVAC_RL:
                novo_tamanho = int(np.round(self.tamanho_inicial - progresso * (self.tamanho_inicial - self.tamanho_minimo)))
                novo_tamanho = max(self.tamanho_minimo, novo_tamanho)
                
                if novo_tamanho < self.tamanho_enxame:
                    indices_melhores = cp.argsort(self.pbest_fitness)[:novo_tamanho]
                    
                    self.posicoes = self.posicoes[indices_melhores]
                    self.velocidades = self.velocidades[indices_melhores]
                    self.pbests = self.pbests[indices_melhores]
                    self.pbest_fitness = self.pbest_fitness[indices_melhores]
                    
                    self.tamanho_enxame = novo_tamanho
            # =========================================================

            epsilon_atual = 0.4 - (0.4 - 0.02) * progresso
            
            topologia = 0 
            w_atual, c1_atual, c2_atual = self.w, self.c1, self.c2
            estado_atual = 0
            acao_escolhida = 0
            
            if self.modo_execucao == ModoPSO.CANONICO:
                pass 
                
            elif self.modo_execucao == ModoPSO.ESTATICO_RL:
                estado_atual = agente.discretizar_estado(erro_anterior, melhor_erro_atual)
                acao_escolhida = agente.selecionar_acao(estado_atual, epsilon_atual)
                topologia = acao_escolhida % 3
                
            # CORREÇÃO 1: Adicionamos os modos 4 e 5 na lista de verificação!
            elif self.modo_execucao in [ModoPSO.TVAC_RL, ModoPSO.LPSR_TVAC_RL, ModoPSO.RL_CINEMATICO, ModoPSO.TVAC_PURO]:
                
                w_base  = 0.9 - (0.9 - 0.4) * progresso
                c1_base = 2.5 - (2.5 - 0.5) * progresso
                c2_base = 0.5 + (2.5 - 0.5) * progresso

                # O TVAC Puro atua sozinho, sem agente
                if self.modo_execucao == ModoPSO.TVAC_PURO:
                    topologia = 0
                    w_atual, c1_atual, c2_atual = w_base, c1_base, c2_base
                    acao_escolhida = 0 # Valor dummy inofensivo
                    
                else:
                    estado_atual = agente.discretizar_estado(erro_anterior, melhor_erro_atual)
                    acao_escolhida = agente.selecionar_acao(estado_atual, epsilon_atual)
                    
                    if progresso > 0.75:
                        acao_escolhida = 1 
                    
                    delta_w = 0.10
                    delta_c = 0.25
                    
                    # CONTROLE RESIDUAL ORIGINAL
                    if acao_escolhida == 0:    
                        topologia = 0; w_atual = w_base + delta_w; c1_atual = c1_base + delta_c; c2_atual = c2_base - delta_c
                    elif acao_escolhida == 1:  
                        topologia = 0; w_atual = w_base - delta_w; c1_atual = c1_base - delta_c; c2_atual = c2_base + delta_c
                    elif acao_escolhida == 2:  
                        topologia = 1; w_atual, c1_atual, c2_atual = w_base, c1_base, c2_base
                    elif acao_escolhida == 3:  
                        topologia = 2; w_atual = w_base + delta_w; c1_atual = c1_base + delta_c; c2_atual = c2_base - delta_c

                    # Trava ortogonal do estudo de ablação
                    if self.modo_execucao == ModoPSO.RL_CINEMATICO:
                        topologia = 0

                    w_atual = max(0.4, min(0.99, w_atual))
                    c1_atual = max(0.5, min(2.5, c1_atual))
                    c2_atual = max(0.5, min(2.5, c2_atual))

            lideres = self._obter_lideres(topologia)
            r1 = self.rng.random((self.tamanho_enxame, self.D))
            r2 = self.rng.random((self.tamanho_enxame, self.D))
            
            self.velocidades = (w_atual * self.velocidades + 
                                c1_atual * r1 * (self.pbests - self.posicoes) + 
                                c2_atual * r2 * (lideres - self.posicoes))
            
            self.velocidades = cp.clip(self.velocidades, -self.Vmax, self.Vmax)
            self.posicoes += self.velocidades
            
            ultrapassou_max = self.posicoes > 100.0
            ultrapassou_min = self.posicoes < -100.0
            self.posicoes = cp.clip(self.posicoes, -100.0, 100.0)
            
            rebote = ultrapassou_max | ultrapassou_min
            self.velocidades = cp.where(rebote, self.velocidades * -0.5, self.velocidades)
            
            centro_massa = cp.mean(self.posicoes, axis=0)
            diversidade_atual = float(cp.mean(cp.linalg.norm(self.posicoes - centro_massa, axis=1)))
            div_anterior = div_salva
            
            qtd_melhorou = 0
            
            if self.avaliacoes_consumidas < self.max_avaliacoes:
                fitness_atual = AvaliadorCEC.avaliar(self.func_id, self.posicoes)
                self.avaliacoes_consumidas += self.tamanho_enxame
                
                melhorou = fitness_atual < self.pbest_fitness
                qtd_melhorou = int(cp.sum(melhorou))  
                
                self.pbest_fitness = cp.where(melhorou, fitness_atual, self.pbest_fitness)
                self.pbests = cp.where(melhorou[:, cp.newaxis], self.posicoes, self.pbests)
            
            erro_anterior = melhor_erro_atual
            melhor_erro_atual = float(cp.min(self.pbest_fitness)) - self.otimo_global
            historico_erro.append(melhor_erro_atual)
            
            # CORREÇÃO 2: Impedimos o TVAC_PURO de tentar aprender (evita quebrar a matriz Q)
            if self.modo_execucao not in [ModoPSO.CANONICO, ModoPSO.TVAC_PURO]:
                proximo_estado = agente.discretizar_estado(erro_anterior, melhor_erro_atual)
                
                recompensa = agente.calcular_recompensa(
                    erro_anterior, melhor_erro_atual, 
                    qtd_melhorou, self.tamanho_enxame, 
                    div_anterior, diversidade_atual, 
                    modo=2 
                )
                agente.atualizar_q(estado_atual, acao_escolhida, recompensa, proximo_estado)
            
            div_salva = diversidade_atual
                
                
        return historico_erro