import numpy as np

class AgenteRL:
    def __init__(self, seed: int = 19937):
        # Substitui o std::mt19937, garantindo reprodutibilidade estrita
        self.rng = np.random.default_rng(seed)
        
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2
        self.num_estados = 4
        self.num_acoes = 3
        
        # Inicializa a Q-Table (4x3) com valores aleatórios entre [-0.1, 0.1)
        self.q_table = self.rng.uniform(-0.1, 0.1, (self.num_estados, self.num_acoes))

    def discretizar_estado(self, erro_anterior: float, erro_atual: float) -> int:
        if erro_anterior <= 1e-12:
            return 3  # Blindagem contra subnormais e zero
            
        delta = (erro_anterior - erro_atual) / erro_anterior
        
        if delta > 0.05:
            return 0  # Decaimento acentuado
        elif delta > 0.01:
            return 1  # Decaimento moderado
        elif delta > 0.0:
            return 2  # Decaimento suave
        return 3      # Estagnação/platô

    def selecionar_acao(self, estado_atual: int) -> int:
        # Exploração (epsilon)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_acoes))
        
        # Explotação (ação com maior valor Q no estado atual)
        return int(np.argmax(self.q_table[estado_atual]))

    def atualizar_q(self, estado_atual: int, acao: int, recompensa: float, proximo_estado: int):
        max_q_proximo = np.max(self.q_table[proximo_estado])
        self.q_table[estado_atual, acao] += self.alpha * (
            recompensa + self.gamma * max_q_proximo - self.q_table[estado_atual, acao]
        )

    def calcular_recompensa(self, estado_atual: int, proximo_estado: int, erro: float) -> float:
        if erro < 1e-8:
            return 100.0  # Critério de corte atingido
        if estado_atual == 3 and (proximo_estado == 0 or proximo_estado == 1):
            return 10.0   # Recompensa por romper inércia
        if proximo_estado == 3:
            return -5.0   # Penalidade por estagnação
        return -1.0       # Custo temporal por transição