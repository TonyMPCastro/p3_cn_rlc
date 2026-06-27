# =============================================================================
# Módulo: Visualização
# Descrição: Classe para geração de gráficos
# =============================================================================

from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt

# Configura o backend para não precisar de tela
matplotlib.use("Agg")

from .circuito import CircuitoRLC
from .solucao_analitica import SolucaoAnalitica
from .metodos_numericos import MetodoEuler, MetodoRK4
from .analise_erros import AnalisadorErros


class Visualizador:
    """
    Gera gráficos para análise da simulação do circuito RLC.
    """
    
    # Cores acadêmicas padrão
    COR_ANALITICO = "black"
    COR_EULER = "#D32F2F"  # Vermelho
    COR_RK4 = "#1976D2"    # Azul
    
    @staticmethod
    def plotar_comparacao(
        t: List[float],
        V_analitico: List[float],
        V_euler: List[float],
        V_rk4: List[float],
        caminho_saida: str = "artigo/comparacao_metodos.png"
    ) -> None:
        """
        Gera gráfico de comparação dos três métodos (visão geral + zoom).
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # --- Painel 1: Visão Geral ---
        ax1.plot(t, V_analitico, label="Solução Analítica",
                color=Visualizador.COR_ANALITICO, linewidth=1.5, zorder=1)
        ax1.plot(t, V_euler, label="Método de Euler",
                color=Visualizador.COR_EULER, linestyle="--", linewidth=1.5, zorder=2)
        ax1.plot(t, V_rk4, label="Método RK4",
                color=Visualizador.COR_RK4, linestyle="-.", linewidth=1.5, zorder=3)
        
        ax1.set_xlabel("Tempo $t$ (s)", fontsize=12, fontfamily="serif")
        ax1.set_ylabel("Tensão $V_C(t)$ (V)", fontsize=12, fontfamily="serif")
        ax1.set_title("Evolução Dinâmica da Tensão (Geral)", fontsize=13, fontfamily="serif")
        ax1.legend(loc="upper right", fontsize=10, edgecolor="black")
        ax1.grid(True, alpha=0.3, linestyle="--")
        
        # Marcar região de zoom
        x1_z, x2_z = 0.002, 0.007
        y1_z, y2_z = -3.5, 0.5
        ax1.axvspan(x1_z, x2_z, color='gray', alpha=0.15)
        
        # --- Painel 2: Zoom ---
        ax2.plot(t, V_analitico, color=Visualizador.COR_ANALITICO, linewidth=1.5, zorder=1)
        ax2.plot(t, V_euler, color=Visualizador.COR_EULER, linestyle="--",
                marker="o", markersize=4, zorder=2)
        ax2.plot(t, V_rk4, color=Visualizador.COR_RK4, linestyle="-.",
                marker="s", markersize=3, zorder=3)
        
        ax2.set_xlim(x1_z, x2_z)
        ax2.set_ylim(y1_z, y2_z)
        ax2.set_xlabel("Tempo $t$ (s)", fontsize=12, fontfamily="serif")
        ax2.set_title(r"Zoom (Destaque do Truncamento Local)", fontsize=13, fontfamily="serif")
        ax2.grid(True, alpha=0.3, linestyle="--")
        
        # Formatação dos eixos
        for ax in [ax1, ax2]:
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
        
        plt.tight_layout()
        plt.savefig(caminho_saida, dpi=300)
        plt.close()
        print(f"  ✓ Gráfico salvo em: {caminho_saida}")
    
    @staticmethod
    def plotar_erro_absoluto(
        t: List[float],
        erros_euler: List[float],
        erros_rk4: List[float],
        C: float,
        caminho_saida: str = "artigo/erro_absoluto.png"
    ) -> None:
        """
        Gera gráfico de erro absoluto (escala logarítmica).
        """
        # Converter para tensão
        erro_V_euler = [e / C for e in erros_euler]
        erro_V_rk4 = [e / C for e in erros_rk4]
        
        # Evitar log(0)
        eps = 1e-16
        erro_V_euler = [max(e, eps) for e in erro_V_euler]
        erro_V_rk4 = [max(e, eps) for e in erro_V_rk4]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(t, erro_V_euler, label="Erro Absoluto - Euler",
               color=Visualizador.COR_EULER, linewidth=1.5, linestyle="-")
        ax.plot(t, erro_V_rk4, label="Erro Absoluto - RK4",
               color=Visualizador.COR_RK4, linewidth=1.5, linestyle="--")
        
        ax.set_yscale("log")
        ax.set_xlabel("Tempo $t$ (s)", fontsize=12, fontfamily="serif")
        ax.set_ylabel("Erro Numérico Global $|V_{num} - V_{exato}|$ (V)",
                     fontsize=12, fontfamily="serif")
        ax.set_title("Evolução do Erro de Truncamento no Tempo",
                    fontsize=14, fontfamily="serif", fontweight="bold")
        
        ax.legend(loc="upper right", fontsize=11, frameon=True, edgecolor="black")
        ax.grid(True, which="major", linestyle="-", alpha=0.3, color="gray")
        ax.grid(True, which="minor", linestyle=":", alpha=0.2, color="gray")
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        plt.savefig(caminho_saida, dpi=300)
        plt.close()
        print(f"  ✓ Gráfico salvo em: {caminho_saida}")
    
    @staticmethod
    def plotar_convergencia(
        circuito: CircuitoRLC,
        t_final: float,
        caminho_saida: str = "artigo/convergencia.png"
    ) -> None:
        """
        Gera gráfico de análise de convergência (erro máximo vs número de passos).
        """
        lista_n_steps = [10, 20, 50, 100, 200, 500, 1000]
        erros_max_euler = []
        erros_max_rk4 = []
        
        # Calcular erros para diferentes resoluções
        sol_analitica = SolucaoAnalitica(circuito)
        metodo_euler = MetodoEuler(circuito)
        metodo_rk4 = MetodoRK4(circuito)
        
        for n in lista_n_steps:
            q_analitico, _ = sol_analitica.calcular(t_final, n)
            q_euler, _, _ = metodo_euler.simular(t_final, n)
            q_rk4, _, _ = metodo_rk4.simular(t_final, n)
            
            # Calcular erros
            _, erro_euler = AnalisadorErros.calcular_erro_absoluto(q_euler, q_analitico)
            _, erro_rk4 = AnalisadorErros.calcular_erro_absoluto(q_rk4, q_analitico)
            
            erros_max_euler.append(erro_euler / circuito.C)
            erros_max_rk4.append(erro_rk4 / circuito.C)
        
        # Plotar
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(lista_n_steps, erros_max_euler, marker="o", markersize=6,
                 color=Visualizador.COR_EULER, label="Método de Euler", linewidth=1.5)
        ax.loglog(lista_n_steps, erros_max_rk4, marker="s", markersize=6,
                 color=Visualizador.COR_RK4, label="Método RK4", linewidth=1.5)
        
        # Triângulo O(h) para Euler
        n1, n2 = lista_n_steps[3], lista_n_steps[4]
        e1, e2 = erros_max_euler[3], erros_max_euler[4]
        ax.plot([n1, n2], [e1, e1], color="black", linewidth=1.0)
        ax.plot([n2, n2], [e1, e2], color="black", linewidth=1.0)
        ax.text(n2 * 1.1, e1, r'$\mathcal{O}(h)$', fontsize=12, fontfamily="serif",
               verticalalignment='center')
        
        # Triângulo O(h^4) para RK4
        nr1, nr2 = lista_n_steps[1], lista_n_steps[2]
        er1, er2 = erros_max_rk4[1], erros_max_rk4[2]
        ax.plot([nr1, nr2], [er1, er1], color="black", linewidth=1.0)
        ax.plot([nr2, nr2], [er1, er2], color="black", linewidth=1.0)
        ax.text(nr2 * 1.1, er1, r'$\mathcal{O}(h^4)$', fontsize=12, fontfamily="serif",
               verticalalignment='center')
        
        ax.set_xlabel("Número de Passos $N$", fontsize=12, fontfamily="serif")
        ax.set_ylabel("Erro Máximo Global $E_{max}$ (V)", fontsize=12, fontfamily="serif")
        ax.set_title("Análise de Convergência (Ordem Global)",
                    fontsize=14, fontfamily="serif", fontweight="bold")
        
        ax.legend(loc="lower left", fontsize=11, frameon=True, edgecolor="black")
        ax.grid(True, which="both", linestyle=":", alpha=0.5, color="gray")
        
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        plt.savefig(caminho_saida, dpi=300)
        plt.close()
        print(f"  ✓ Gráfico salvo em: {caminho_saida}")
