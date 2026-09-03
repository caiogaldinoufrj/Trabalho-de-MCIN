import os
import numpy as np

# Tenta carregar o CuPy para processamento na Placa de Vídeo. 
# Caso esteja testando em uma máquina sem GPU local, o fallback para NumPy garante que o código não quebre.
try:
    import cupy as cp
except ImportError:
    print("Aviso: CuPy não encontrado. Operando via CPU com NumPy estrito.")
    import numpy as cp

class TransformadorCEC:
    def __init__(self, D: int, funcoes_alvo: list):
        self.D = D
        self.shifts = {}
        self.matrizes = {}
        
        for func in funcoes_alvo:
            # Carrega Shift
            path_shift = f"input_data/shift_data_{func}.txt"
            if not os.path.exists(path_shift):
                raise FileNotFoundError(f"Erro fatal: {path_shift} ausente.")
            with open(path_shift, 'r') as f:
                dados = [float(x) for x in f.read().split()]
                # Cria o tensor diretamente na VRAM (Memória de Vídeo)
                self.shifts[func] = cp.array(dados[:D])
            
            # Carrega Matriz de Rotação
            path_mat = f"input_data/M_{func}_D{D}.txt"
            if not os.path.exists(path_mat):
                raise FileNotFoundError(f"Erro fatal: {path_mat} ausente.")
            with open(path_mat, 'r') as f:
                dados = [float(x) for x in f.read().split()]
                # Reshape para DxD e envia para a VRAM
                self.matrizes[func] = cp.array(dados[:D*D]).reshape((D, D))

    def transformar(self, func_id: int, x: cp.ndarray, escala: float) -> cp.ndarray:
        """
        Recebe x como uma matriz (tamanho_enxame, D).
        Aplica o shift e a multiplicação matricial em lote na GPU.
        """
        x_shift = (x - self.shifts[func_id]) * escala
        # Produto matricial vetorizado: Z = X_shift * M^T
        z = cp.matmul(x_shift, self.matrizes[func_id].T)
        return z

class AvaliadorCEC:
    transformador = None

    @classmethod
    def inicializar_transformador(cls, D: int, funcoes_alvo: list):
        cls.transformador = TransformadorCEC(D, funcoes_alvo)

    @classmethod
    def avaliar(cls, func_id: int, x: cp.ndarray) -> cp.ndarray:
        """
        Calcula o fitness de TODAS as partículas simultaneamente.
        Retorna um vetor (tamanho_enxame, ) com os erros.
        """
        D = x.shape[1]
        otimo_global = func_id * 100.0

        if func_id == 2:
            z = cls.transformador.transformar(func_id, x, 1.0)
            fitness = z[:, 0]**2 + 1e6 * cp.sum(z[:, 1:]**2, axis=1)
            
        elif func_id == 4:
            z = cls.transformador.transformar(func_id, x, 2.048 / 100.0)
            z = z + 1.0
            t1 = z[:, :-1]**2 - z[:, 1:]
            t2 = z[:, :-1] - 1.0
            fitness = cp.sum(100.0 * t1**2 + t2**2, axis=1)
            
        elif func_id == 6:
            z = cls.transformador.transformar(func_id, x, 0.5 / 100.0)
            k = cp.arange(21)
            a, b = 0.5, 3.0
            ak = a**k
            bk = b**k
            
            # Constante escalar
            sum2 = cp.sum(ak * cp.cos(2.0 * cp.pi * bk * 0.5))
            
            # Broadcasting mágico tridimensional: shape (pop, D, 1) * (21,) -> (pop, D, 21)
            z_ext = z[:, :, cp.newaxis] 
            termos = ak * cp.cos(2.0 * cp.pi * bk * (z_ext + 0.5))
            sum1 = cp.sum(termos, axis=2) # Soma na dimensão k
            fitness = cp.sum(sum1, axis=1) - D * sum2 # Soma na dimensão D
            
        elif func_id == 7:
            z = cls.transformador.transformar(func_id, x, 600.0 / 100.0)
            sum_part = cp.sum(z**2 / 4000.0, axis=1)
            indices = cp.sqrt(cp.arange(1, D + 1))
            prod_part = cp.prod(cp.cos(z / indices), axis=1)
            fitness = sum_part - prod_part + 1.0
            
        elif func_id == 9:
            z = cls.transformador.transformar(func_id, x, 5.12 / 100.0)
            fitness = cp.sum(z**2 - 10.0 * cp.cos(2.0 * cp.pi * z) + 10.0, axis=1)
            
        elif func_id == 13:
            z = cls.transformador.transformar(func_id, x, 5.0 / 100.0)
            z = z - 1.0
            sum_sq = cp.sum(z**2, axis=1)
            sum_z = cp.sum(z, axis=1)
            fitness = cp.power(cp.abs(sum_sq - D), 0.25) + (0.5 * sum_sq + sum_z) / D + 0.5
            
        else:
            raise ValueError(f"Funcao {func_id} não suportada neste bloco de ablação.")

        return fitness + otimo_global