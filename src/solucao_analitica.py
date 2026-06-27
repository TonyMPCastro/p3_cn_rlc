# =============================================================================
# Módulo: Solução Analítica
# Descrição: Calcula a solução exata via Transformada de Laplace
# =============================================================================

import math
from typing import Tuple, List
import numpy as np

from .circuito import CircuitoRLC


class SolucaoAnalitica:
    """
    Calcula a solução exata da EDO do circuito RLC usando Transformada de Laplace.
    
    A EDO do circuito é:
        L * q''(t) + R * q'(t) + (1/C) * q(t) = 0
    
    Com condições iniciais:
        q(0) = C * V0   (carga inicial)
        q'(0) = 0        (corrente inicial nula)
    """
    
    def __init__(self, circuito: CircuitoRLC):
        """Inicializa com um circuito RLC."""
        self.circuito = circuito
    
    def calcular(self, t_final: float, n_steps: int) -> Tuple[List[float], List[float]]:
        """
        Calcula a solução analítica em tempos discretos.
        
        Parâmetros:
            t_final: Tempo final da simulação (segundos)
            n_steps: Número de passos de tempo
        
        Retorna:
            Tupla (q, t) com cargas e tempos
        """
        # Validação
        if t_final <= 0:
            raise ValueError("t_final deve ser positivo.")
        if n_steps <= 0:
            raise ValueError("n_steps deve ser positivo.")
        
        # Criar vetor de tempo
        t = np.linspace(0.0, t_final, n_steps + 1)
        
        # Calcular carga em cada instante
        q = []
        for ti in t:
            q_ti = self._avaliar_em_tempo(ti)
            q.append(q_ti)
        
        return q, t.tolist()
    
    def _avaliar_em_tempo(self, t: float) -> float:
        """Avalia q(t) em um instante específico."""
        c = self.circuito
        q0 = c.q0
        alpha = c.alpha
        omega0 = c.omega0
        delta = c.discriminante
        
        if abs(delta) < 1e-14:
            # --- Criticamente amortecido ---
            return q0 * math.exp(-alpha * t) * (1.0 + alpha * t)
        
        elif delta > 0:
            # --- Superamortecido ---
            s1 = -alpha + math.sqrt(delta)
            s2 = -alpha - math.sqrt(delta)
            return q0 * (s2 * math.exp(s1 * t) - s1 * math.exp(s2 * t)) / (s2 - s1)
        
        else:
            # --- Subamortecido ---
            omega_d = math.sqrt(-delta)
            return q0 * math.exp(-alpha * t) * (
                math.cos(omega_d * t) + (alpha / omega_d) * math.sin(omega_d * t)
            )
    
    def tensao(self, t_final: float, n_steps: int) -> Tuple[List[float], List[float]]:
        """
        Retorna a tensão V(t) = q(t) / C em vez de carga.
        
        Retorna:
            Tupla (V, t) com tensões e tempos
        """
        q, t = self.calcular(t_final, n_steps)
        V = [q_i / self.circuito.C for q_i in q]
        return V, t
