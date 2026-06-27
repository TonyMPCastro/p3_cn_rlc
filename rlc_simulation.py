# =============================================================================
# Simulação Numérica de um Circuito RLC Série
# Descarga de Capacitores em Sensores IoT
#
# Versão Modular com Classes
#
# Autores: Antonio Marcos, Felipe Carneiro, Pedro Kaic
# Disciplina: Cálculo Numérico - UFMA
# =============================================================================

import math
from typing import List

from src.circuito import CircuitoRLC
from src.solucao_analitica import SolucaoAnalitica
from src.metodos_numericos import MetodoEuler, MetodoRK4
from src.analise_erros import AnalisadorErros
from src.visualizacao import Visualizador


class SimuladorRLC:
    """
    Orquestra a simulação completa do circuito RLC.
    """
    
    def __init__(self, circuito: CircuitoRLC, t_final: float, n_steps: int):
        """
        Inicializa o simulador.
        
        Parâmetros:
            circuito: Instância de CircuitoRLC
            t_final: Tempo final da simulação (segundos)
            n_steps: Número de passos de tempo
        """
        self.circuito = circuito
        self.t_final = t_final
        self.n_steps = n_steps
        
        # Instanciar solvedores
        self.sol_analitica = SolucaoAnalitica(circuito)
        self.euler = MetodoEuler(circuito)
        self.rk4 = MetodoRK4(circuito)
    
    def executar(self) -> dict:
        """
        Executa a simulação completa.
        
        Retorna um dicionário com:
            - 't': lista de tempos
            - 'V_analitico': tensão analítica
            - 'V_euler': tensão Euler
            - 'V_rk4': tensão RK4
            - 'erros_abs_euler': erros absolutos Euler
            - 'erros_abs_rk4': erros absolutos RK4
            - 'resumo_euler': resumo de erros Euler
            - 'resumo_rk4': resumo de erros RK4
        """
        print("\n" + "=" * 70)
        print("  SIMULAÇÃO NUMÉRICA - CIRCUITO RLC SÉRIE")
        print("  Descarga de Capacitor em Sensor IoT")
        print("=" * 70)
        
        # Mostrar parâmetros
        self._mostrar_parametros()
        
        # Executar cálculos
        print("\n--- Executando Simulações ---")
        q_analitico, t = self.sol_analitica.calcular(self.t_final, self.n_steps)
        q_euler, i_euler, _ = self.euler.simular(self.t_final, self.n_steps)
        q_rk4, i_rk4, _ = self.rk4.simular(self.t_final, self.n_steps)
        
        # Converter para tensão: V = q / C
        V_analitico = [q / self.circuito.C for q in q_analitico]
        V_euler = [q / self.circuito.C for q in q_euler]
        V_rk4 = [q / self.circuito.C for q in q_rk4]
        
        # Calcular erros
        print("\n--- Calculando Erros ---")
        erros_abs_euler, _ = AnalisadorErros.calcular_erro_absoluto(q_euler, q_analitico)
        erros_abs_rk4, _ = AnalisadorErros.calcular_erro_absoluto(q_rk4, q_analitico)
        
        resumo_euler = AnalisadorErros.resumo_erros(q_euler, q_analitico, "Método de Euler")
        resumo_rk4 = AnalisadorErros.resumo_erros(q_rk4, q_analitico, "Método RK4")
        
        print(f"  Erro máximo Euler: {resumo_euler['erro_max']/self.circuito.C:.6e} V")
        print(f"  Erro máximo RK4:   {resumo_rk4['erro_max']/self.circuito.C:.6e} V")
        
        # Mostrar tabela
        self._mostrar_tabela(t, V_analitico, V_euler, V_rk4)
        
        # Gerar gráficos
        print("\n--- Gerando Gráficos ---")
        Visualizador.plotar_comparacao(t, V_analitico, V_euler, V_rk4)
        Visualizador.plotar_erro_absoluto(t, erros_abs_euler, erros_abs_rk4, self.circuito.C)
        Visualizador.plotar_convergencia(self.circuito, self.t_final)
        
        # Mostrar conclusão
        self._mostrar_conclusao(resumo_euler, resumo_rk4)
        
        # Retornar resultados
        return {
            't': t,
            'V_analitico': V_analitico,
            'V_euler': V_euler,
            'V_rk4': V_rk4,
            'erros_abs_euler': erros_abs_euler,
            'erros_abs_rk4': erros_abs_rk4,
            'resumo_euler': resumo_euler,
            'resumo_rk4': resumo_rk4
        }
    
    def _mostrar_parametros(self) -> None:
        """Exibe os parâmetros do circuito."""
        c = self.circuito
        print("\n--- Parâmetros do Circuito ---")
        print(f"  R  = {c.R} Ohms (resistência)")
        print(f"  L  = {c.L} H = {c.L*1000:.1f} mH (indutância)")
        print(f"  C  = {c.C} F = {c.C*1e6:.0f} µF (capacitância)")
        print(f"  V0 = {c.V0} V (tensão inicial)")
        print(f"  Tempo final = {self.t_final*1000:.1f} ms")
        print(f"  Passos = {self.n_steps}")
        print(f"  h (tamanho do passo) = {self.t_final/self.n_steps*1000:.4f} ms")
        print(f"\n  α (amortecimento) = {c.alpha:.2f}")
        print(f"  ω₀ (freq. natural) = {c.omega0:.2f}")
        print(f"  Regime: {c.regime.upper().replace('_', ' ')}")
    
    def _mostrar_tabela(self, t: List[float], V_an: List[float],
                       V_eu: List[float], V_rk: List[float]) -> None:
        """Exibe tabela de resultados em pontos selecionados."""
        print("\n--- Tabela de Resultados (pontos selecionados) ---")
        print("-" * 100)
        print(f"{'t (ms)':>8} | {'V_Analítico (V)':>16} | {'V_Euler (V)':>13} | "
              f"{'V_RK4 (V)':>11} | {'Erro Euler (V)':>15} | {'Erro RK4 (V)':>13}")
        print("-" * 100)
        
        # Selecionar pontos: início, 1/4, 1/2, 3/4, final
        indices = [0, self.n_steps // 4, self.n_steps // 2,
                  3 * self.n_steps // 4, self.n_steps]
        
        for idx in indices:
            t_ms = t[idx] * 1000
            err_euler = abs(V_eu[idx] - V_an[idx])
            err_rk4 = abs(V_rk[idx] - V_an[idx])
            
            print(f"{t_ms:>8.3f} | {V_an[idx]:>16.6f} | {V_eu[idx]:>13.6f} | "
                  f"{V_rk[idx]:>11.6f} | {err_euler:>15.6e} | {err_rk4:>13.6e}")
        
        print("-" * 100)
    
    def _mostrar_conclusao(self, resumo_euler: dict, resumo_rk4: dict) -> None:
        """Exibe conclusão da simulação."""
        print("\n--- Conclusão ---")
        
        if resumo_rk4['erro_max'] > 1e-14:
            razao = resumo_euler['erro_max'] / resumo_rk4['erro_max']
            print(f"  O RK4 foi {razao:.0f}x mais preciso que o Euler para {self.n_steps} passos.")
        else:
            print(f"  O RK4 atingiu precisão numérica máxima (~1e-16).")
        
        print("  ✓ Simulação concluída com sucesso!")
        print("=" * 70)


def main():
    """Função principal."""
    
    # Definir parâmetros do circuito - Valores ideais para testes didáticos (subamortecido)
    # Valores típicos para um sensor IoT com supercapacitor
    R = 2.0        # Resistência total do circuito (Ohms)
    L = 0.01       # Indutância parasita das trilhas (Henrys) = 10 mH
    C = 1e-4       # Capacitância do supercapacitor (Farads) = 100 uF
    V0 = 3.3       # Tensão inicial de operação (Volts)
    t_final = 0.015 # Tempo final da simulação (segundos) = 15 ms
    n_steps = 60   # Número de passos de tempo
    
    # Criar circuito
    circuito = CircuitoRLC(R=R, L=L, C=C, V0=V0)
    
    # Executar simulação
    simulador = SimuladorRLC(circuito, t_final, n_steps)
    resultados = simulador.executar()
    
    return resultados


if __name__ == "__main__":
    main()
