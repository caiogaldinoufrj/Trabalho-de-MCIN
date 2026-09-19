import numpy as np

class AgenteRL:
    def __init__(self, seed: int = 19937):
        self.rng = np.random.default_rng(seed)
        
        self.alpha = 0.05
        self.gamma = 0.9
        self.epsilon = 0.2
        self.num_estados = 4
        self.num_acoes = 4  # <-- Q-Table 4x4
        
        # INICIALIZAÇÃO COM VIÉS ESPECIALISTA (Warm-Start)
        # Ações: [0: Estrela+Acelerador, 1: Estrela+Freio, 2: Anel+Neutro, 3: Aleatória+Fuga]
        self.q_table = np.array([
            [ 0.5,  1.0,  0.0, -0.5],  # Est 0 (Queda rápida): Prefere Estrela+Freio (Deslize seguro)
            [ 0.8,  0.5,  0.2, -0.2],  # Est 1 (Queda moderada): Prefere Estrela+Acelerador
            [ 0.0,  0.0,  1.0,  0.5],  # Est 2 (Queda suave): Aciona o Anel para contornar
            [-0.5, -0.5,  0.5,  1.0]   # Est 3 (Platô): Aciona Aleatória+Fuga urgente!
        ]) + self.rng.uniform(-0.05, 0.05, (self.num_estados, self.num_acoes))

    def discretizar_estado(self, erro_anterior: float, erro_atual: float) -> int:
        if erro_anterior <= 1e-12: return 3
        delta = (erro_anterior - erro_atual) / erro_anterior
        if delta > 0.05: return 0 
        elif delta > 0.01: return 1  
        elif delta > 0.0: return 2  
        return 3      

    def selecionar_acao(self, estado_atual: int, epsilon_atual: float = None) -> int:
        eps = self.epsilon if epsilon_atual is None else epsilon_atual
        if self.rng.random() < eps:
            return int(self.rng.integers(0, self.num_acoes))
        return int(np.argmax(self.q_table[estado_atual]))

    def atualizar_q(self, estado_atual: int, acao: int, recompensa: float, proximo_estado: int):
        max_q_proximo = np.max(self.q_table[proximo_estado])
        self.q_table[estado_atual, acao] += self.alpha * (
            recompensa + self.gamma * max_q_proximo - self.q_table[estado_atual, acao]
        )

    def calcular_recompensa(self, erro_anterior: float, erro_atual: float, 
                            qtd_melhorou: int, tamanho_pop: int, 
                            diversidade_anterior: float, diversidade_atual: float, 
                            modo: int = 2) -> float:
        
        # 1. Condição de Sucesso Universal
        if erro_atual < 1e-8:
            return 1.0  
            
        recompensa = 0.0
        
        # MODO 1: FIR (Fitness Improvement Rate Matemático)
        if modo == 1:
            if erro_anterior > 0:
                recompensa = (erro_anterior - erro_atual) / erro_anterior
            else:
                recompensa = 0.0
                
        # MODO 2: Melhoria Populacional (Proporção do enxame que achou Pbest novo)
        elif modo == 2:
            recompensa = qtd_melhorou / tamanho_pop
            
        # MODO 3: Multiobjetivo (FIR + Diversidade)
        elif modo == 3:
            fir = (erro_anterior - erro_atual) / erro_anterior if erro_anterior > 0 else 0.0
            
            delta_div = 0.0
            if diversidade_anterior > 0:
                delta_div = (diversidade_atual - diversidade_anterior) / diversidade_anterior
            
            if fir > 1e-5:
                recompensa = fir
            else:
                recompensa = delta_div 
                
        # MODO 4: A Híbrida Completa 
        elif modo == 4:
            fir = (erro_anterior - erro_atual) / erro_anterior if erro_anterior > 0 else 0.0
            pop_rate = qtd_melhorou / tamanho_pop
            recompensa = 0.5 * fir + 0.5 * pop_rate
            
        return float(np.clip(recompensa, -1.0, 1.0))