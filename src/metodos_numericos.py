# =============================================================================
# Módulo: Métodos Numéricos
# Descrição: Classes para Método de Euler e Runge-Kutta RK4
# =============================================================================

from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np

from .circuito import CircuitoRLC


class MetodoNumerico(ABC):
    """Classe abstrata para métodos numéricos."""
    
    def __init__(self, circuito: CircuitoRLC):
        """Inicializa com um circuito RLC."""
        self.circuito = circuito
    
    @abstractmethod
    def simular(self, t_final: float, n_steps: int) -> Tuple[List[float], List[float], List[float]]:
        """
        Simula a evolução do circuito.
        
        Retorna:
            Tupla (q, i, t) com carga, corrente e tempos
        """
        pass


class MetodoEuler(MetodoNumerico):
    """
    Método de Euler explícito para integração numérica.
    
    Ordem de convergência: O(h)
    """
    
    def simular(self, t_final: float, n_steps: int) -> Tuple[List[float], List[float], List[float]]:
        """
        Simula usando Método de Euler.
        
        Converte a EDO de 2ª ordem em sistema de 1ª ordem:
            dq/dt = i
            di/dt = -(1/LC)*q - (R/L)*i
        """
        # Validações
        if t_final <= 0:
            raise ValueError("t_final deve ser positivo.")
        if n_steps <= 0:
            raise ValueError("n_steps deve ser positivo.")
        
        c = self.circuito
        h = t_final / n_steps
        
        # Condições iniciais
        q_list = [c.q0]
        i_list = [0.0]
        t_list = [0.0]
        
        # Simulação passo a passo
        for n in range(n_steps):
            q_prev = q_list[-1]
            i_prev = i_list[-1]
            
            # Derivadas: dq/dt = i, di/dt = -(1/LC)*q - (R/L)*i
            dq_dt = i_prev
            di_dt = -(1.0 / (c.L * c.C)) * q_prev - (c.R / c.L) * i_prev
            
            # Avançar um passo: y_{n+1} = y_n + h * f(y_n)
            q_next = q_prev + h * dq_dt
            i_next = i_prev + h * di_dt
            
            q_list.append(q_next)
            i_list.append(i_next)
            t_list.append((n + 1) * h)
        
        return q_list, i_list, t_list


class MetodoRK4(MetodoNumerico):
    """
    Método de Runge-Kutta de 4ª ordem para integração numérica.
    
    Ordem de convergência: O(h^4)
    """
    
    def simular(self, t_final: float, n_steps: int) -> Tuple[List[float], List[float], List[float]]:
        """
        Simula usando Método RK4.
        
        Usa 4 avaliações da derivada para melhor precisão.
        """
        # Validações
        if t_final <= 0:
            raise ValueError("t_final deve ser positivo.")
        if n_steps <= 0:
            raise ValueError("n_steps deve ser positivo.")
        
        c = self.circuito
        h = t_final / n_steps
        
        # Condições iniciais
        q_list = [c.q0]
        i_list = [0.0]
        t_list = [0.0]
        
        # Simulação passo a passo
        for n in range(n_steps):
            q_prev = q_list[-1]
            i_prev = i_list[-1]
            
            # --- Estágio 1: Inclinação no início do intervalo ---
            k1_q = i_prev
            k1_i = -(1.0 / (c.L * c.C)) * q_prev - (c.R / c.L) * i_prev
            
            # --- Estágio 2: Inclinação no meio, usando k1 ---
            k2_q = i_prev + 0.5 * h * k1_i
            k2_i = (-(1.0 / (c.L * c.C)) * (q_prev + 0.5 * h * k1_q)
                    - (c.R / c.L) * (i_prev + 0.5 * h * k1_i))
            
            # --- Estágio 3: Inclinação no meio, usando k2 ---
            k3_q = i_prev + 0.5 * h * k2_i
            k3_i = (-(1.0 / (c.L * c.C)) * (q_prev + 0.5 * h * k2_q)
                    - (c.R / c.L) * (i_prev + 0.5 * h * k2_i))
            
            # --- Estágio 4: Inclinação no fim do intervalo, usando k3 ---
            k4_q = i_prev + h * k3_i
            k4_i = (-(1.0 / (c.L * c.C)) * (q_prev + h * k3_q)
                    - (c.R / c.L) * (i_prev + h * k3_i))
            
            # --- Média ponderada das 4 inclinações ---
            # x_{n+1} = x_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            q_next = q_prev + (h / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
            i_next = i_prev + (h / 6.0) * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i)
            
            q_list.append(q_next)
            i_list.append(i_next)
            t_list.append((n + 1) * h)
        
        return q_list, i_list, t_list
