# =============================================================================
# Simulação Numérica de um Circuito RLC Série
# Descarga de Capacitores em Sensores IoT
#
# Métodos implementados:
#   1. Solução analítica (Transformada de Laplace)
#   2. Método de Euler explícito
#   3. Método de Runge-Kutta de 4ª ordem (RK4)
#
# Autores: Antonio Marcos, Felipe Carneiro, Pedro Kaic
# Disciplina: Cálculo Numérico - UFMA
# =============================================================================

import math
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configura o backend para não precisar de tela (evita erros em servidores)
matplotlib.use("Agg")


# =====================================================================
# PARTE 1 - SOLUÇÃO ANALÍTICA (TRANSFORMADA DE LAPLACE)
# =====================================================================

def laplace_transform_solution(
    R: float, L: float, C: float, V0: float,
    t_final: float, n_steps: int
) -> Tuple[List[float], List[float]]:
    """
    Calcula a solução EXATA da EDO do circuito RLC usando Transformada de Laplace.

    A EDO do circuito é:
        L * q''(t) + R * q'(t) + (1/C) * q(t) = 0

    Com condições iniciais:
        q(0) = C * V0   (carga inicial no capacitor)
        q'(0) = 0        (corrente inicial nula)

    Parâmetros:
        R       - Resistência em Ohms
        L       - Indutância em Henrys
        C       - Capacitância em Farads
        V0      - Tensão inicial no capacitor em Volts
        t_final - Tempo final da simulação em segundos
        n_steps - Número de passos de tempo

    Retorna:
        q       - Lista com os valores de carga q(t) em cada instante
        t       - Lista com os instantes de tempo
    """
    # Validação dos parâmetros
    if L <= 0 or C <= 0:
        raise ValueError("L and C must be positive.")

    # Passo 1: Calcular a carga inicial
    q0 = C * V0

    # Passo 2: Calcular os parâmetros do circuito
    # alpha = coeficiente de amortecimento
    # omega0 = frequência natural não amortecida
    alpha = R / (2.0 * L)
    omega0 = 1.0 / math.sqrt(L * C)

    # Passo 3: Calcular o discriminante para saber o tipo de amortecimento
    # delta > 0 => superamortecido (raízes reais distintas)
    # delta = 0 => criticamente amortecido (raízes reais iguais)
    # delta < 0 => subamortecido (raízes complexas conjugadas)
    delta = alpha * alpha - omega0 * omega0

    # Passo 4: Criar o vetor de tempo
    t = np.linspace(0.0, t_final, n_steps + 1)

    # Passo 5: Calcular q(t) para cada instante de tempo
    q = []
    for ti in t:
        if abs(delta) < 1e-14:
            # --- Caso criticamente amortecido ---
            # q(t) = q0 * e^(-alpha*t) * (1 + alpha*t)
            q_t = q0 * math.exp(-alpha * ti) * (1.0 + alpha * ti)

        elif delta > 0:
            # --- Caso superamortecido ---
            # Raízes reais distintas: s1 e s2
            s1 = -alpha + math.sqrt(delta)
            s2 = -alpha - math.sqrt(delta)
            # q(t) = q0 * (s2*e^(s1*t) - s1*e^(s2*t)) / (s2 - s1)
            q_t = q0 * (s2 * math.exp(s1 * ti) - s1 * math.exp(s2 * ti)) / (s2 - s1)

        else:
            # --- Caso subamortecido ---
            # Frequência de oscilação amortecida
            omega_d = math.sqrt(-delta)
            # q(t) = q0 * e^(-alpha*t) * [cos(omega_d*t) + (alpha/omega_d)*sin(omega_d*t)]
            q_t = q0 * math.exp(-alpha * ti) * (
                math.cos(omega_d * ti) + (alpha / omega_d) * math.sin(omega_d * ti)
            )

        q.append(q_t)

    return q, t.tolist()


def analytic_solution(
    R: float, L: float, C: float, V0: float,
    t_final: float, n_steps: int
) -> Tuple[List[float], List[float]]:
    """Wrapper de compatibilidade para a solução analítica via Laplace."""
    return laplace_transform_solution(R, L, C, V0, t_final, n_steps)


# =====================================================================
# PARTE 2 - MÉTODOS NUMÉRICOS (EULER E RK4)
# =====================================================================

def simulate_rlc(
    R: float, L: float, C: float, V0: float,
    t_final: float, n_steps: int
) -> Tuple[List[float], List[float], List[float]]:
    """
    Simula a descarga do circuito RLC usando Euler explícito e RK4.

    A EDO de 2ª ordem é convertida em um sistema de 2 EDOs de 1ª ordem:
        dq/dt = i          (a derivada da carga é a corrente)
        di/dt = -(1/LC)*q - (R/L)*i   (vem da Lei de Kirchhoff)

    Parâmetros:
        R, L, C, V0     - Parâmetros do circuito
        t_final          - Tempo final da simulação
        n_steps          - Número de passos de tempo

    Retorna:
        q_euler  - Lista com carga calculada pelo Método de Euler
        q_rk4    - Lista com carga calculada pelo Método RK4
        t        - Lista com os instantes de tempo
    """
    # Validação dos parâmetros
    if L <= 0 or C <= 0:
        raise ValueError("L and C must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    # Passo 1: Definir condições iniciais
    q0 = C * V0       # carga inicial (Coulombs)
    h = t_final / n_steps  # tamanho do passo de tempo

    # Passo 2: Criar o vetor de tempo
    t = np.linspace(0.0, t_final, n_steps + 1)

    # Passo 3: Inicializar vetores de resultado
    # Para o Método de Euler
    q_euler = [q0]     # lista de cargas (Euler)
    i_euler = [0.0]    # lista de correntes (Euler)

    # Para o Método RK4
    q_rk4 = [q0]       # lista de cargas (RK4)
    i_rk4 = [0.0]      # lista de correntes (RK4)

    # Passo 4: Loop principal - avançar no tempo
    for n in range(n_steps):

        # ============================================
        # MÉTODO DE EULER EXPLÍCITO
        # ============================================
        # Pegar os valores atuais
        q_prev = q_euler[-1]
        i_prev = i_euler[-1]

        # Calcular as derivadas no ponto atual
        dq_dt = i_prev                                          # dq/dt = i
        di_dt = -(1.0 / (L * C)) * q_prev - (R / L) * i_prev   # di/dt

        # Avançar um passo: y_{n+1} = y_n + h * f(t_n, y_n)
        q_next_euler = q_prev + h * dq_dt
        i_next_euler = i_prev + h * di_dt

        # Guardar os resultados
        q_euler.append(q_next_euler)
        i_euler.append(i_next_euler)

        # ============================================
        # MÉTODO DE RUNGE-KUTTA DE 4ª ORDEM (RK4)
        # ============================================
        # Pegar os valores atuais
        q_prev_rk4 = q_rk4[-1]
        i_prev_rk4 = i_rk4[-1]

        # --- Estágio 1: inclinação no INÍCIO do intervalo ---
        k1_q = i_prev_rk4
        k1_i = -(1.0 / (L * C)) * q_prev_rk4 - (R / L) * i_prev_rk4

        # --- Estágio 2: inclinação no MEIO, usando k1 ---
        k2_q = i_prev_rk4 + 0.5 * h * k1_i
        k2_i = (-(1.0 / (L * C)) * (q_prev_rk4 + 0.5 * h * k1_q)
                - (R / L) * (i_prev_rk4 + 0.5 * h * k1_i))

        # --- Estágio 3: inclinação no MEIO, usando k2 ---
        k3_q = i_prev_rk4 + 0.5 * h * k2_i
        k3_i = (-(1.0 / (L * C)) * (q_prev_rk4 + 0.5 * h * k2_q)
                - (R / L) * (i_prev_rk4 + 0.5 * h * k2_i))

        # --- Estágio 4: inclinação no FIM do intervalo, usando k3 ---
        k4_q = i_prev_rk4 + h * k3_i
        k4_i = (-(1.0 / (L * C)) * (q_prev_rk4 + h * k3_q)
                - (R / L) * (i_prev_rk4 + h * k3_i))

        # --- Média ponderada das 4 inclinações ---
        # x_{n+1} = x_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        q_next_rk4 = q_prev_rk4 + (h / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        i_next_rk4 = i_prev_rk4 + (h / 6.0) * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i)

        # Guardar os resultados
        q_rk4.append(q_next_rk4)
        i_rk4.append(i_next_rk4)

    return q_euler, q_rk4, t.tolist()


# =====================================================================
# PARTE 3 - CÁLCULO DE ERROS
# =====================================================================

def calcular_erros(q_numerico: List[float], q_analitico: List[float]) -> Tuple[List[float], float]:
    """
    Calcula o erro absoluto em cada ponto e o erro global (máximo).

    Parâmetros:
        q_numerico  - Valores calculados pelo método numérico
        q_analitico - Valores da solução analítica (referência)

    Retorna:
        erros     - Lista com o erro absoluto em cada ponto
        erro_max  - Maior erro absoluto encontrado
    """
    erros = []
    for i in range(len(q_numerico)):
        erro = abs(q_numerico[i] - q_analitico[i])
        erros.append(erro)

    erro_max = max(erros)
    return erros, erro_max


# =====================================================================
# PARTE 4 - GERAÇÃO DE GRÁFICOS
# =====================================================================

def plot_comparacao(t, v_analitico, v_euler, v_rk4, output_path="comparacao_metodos.png"):
    """
    Grafico 1: Comparacao das curvas de tensao V(t) dos 3 metodos.
    Inclui um Inset (Zoom) para destacar a falha grotesca do metodo de Euler.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    # Grafico principal com marcadores para evidenciar os passos discretos
    ax.plot(t, v_analitico, label="Solucao Analitica (Exata)",
             color="black", linewidth=2, zorder=1)
    ax.plot(t, v_euler, label="Metodo de Euler (h largo)",
             color="royalblue", linestyle="--", marker="o", markersize=5, linewidth=1.5, zorder=2)
    ax.plot(t, v_rk4, label="Metodo RK4",
             color="crimson", linestyle=":", marker="s", markersize=4, linewidth=1.5, zorder=3)

    ax.set_xlabel("Tempo (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tensao no capacitor (V)", fontsize=12, fontweight="bold")
    ax.set_title("Dinâmica de Descarga: Solucao Analitica vs Euler vs RK4", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle="--")

    # --- Criando o Zoom (Inset Axes) ---
    # Posicionado no meio-baixo do grafico principal
    axins = ax.inset_axes([0.35, 0.15, 0.45, 0.4])
    axins.plot(t, v_analitico, color="black", linewidth=2)
    axins.plot(t, v_euler, color="royalblue", linestyle="--", marker="o", markersize=5, linewidth=1.5)
    axins.plot(t, v_rk4, color="crimson", linestyle=":", marker="s", markersize=4, linewidth=1.5)

    # Focar na região de 2ms a 7ms, onde ocorre o grande vale de oscilação
    x1, x2 = 0.002, 0.007
    y1, y2 = -3.5, 0.5
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_title("Destaque: Divergencia Numérica do Euler", fontsize=10, fontweight="bold", color="darkred")
    axins.grid(True, alpha=0.3, linestyle=":")
    axins.tick_params(axis='both', which='major', labelsize=8)

    # Desenhar linhas conectando a regiao de zoom ao inset
    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.6, linewidth=1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300) # Salvando em alta resolução
    plt.close()
    print(f"  -> Grafico salvo em: {output_path}")


def plot_erro_absoluto(t, erros_euler, erros_rk4, C, output_path="erro_absoluto.png"):
    """
    Gráfico 2: Erro absoluto da tensão de Euler e RK4 ao longo do tempo.
    Modificado para ser altamente didático e fácil de compreender.
    """
    erro_v_euler = [e / C for e in erros_euler]
    erro_v_rk4_mv = [(e / C) * 1000 for e in erros_rk4] # Convertido para milivolts para facilitar leitura

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Subplot 1: Erro do Euler
    ax1.plot(t, erro_v_euler, color="red", linewidth=2)
    ax1.fill_between(t, 0, erro_v_euler, color="red", alpha=0.2)
    ax1.set_ylabel("Erro Euler (V)\n[Escala de Volts]", fontsize=11, fontweight="bold", color="darkred")
    ax1.set_title("Evolução do Erro Absoluto Numérico", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.4, linestyle="--")

    # Subplot 2: Erro do RK4
    ax2.plot(t, erro_v_rk4_mv, color="green", linewidth=2)
    ax2.fill_between(t, 0, erro_v_rk4_mv, color="green", alpha=0.2)
    ax2.set_xlabel("Tempo (s)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Erro RK4 (mV)\n[Escala de Milivolts]", fontsize=11, fontweight="bold", color="darkgreen")
    ax2.grid(True, alpha=0.4, linestyle="--")
    
    # Caixa de texto explicativa
    texto = "Nota de Leitura: Observe a diferença brutal nas escalas (Eixo Y).\nEnquanto o Euler erra na casa dos Volts, o RK4 se mantém restrito a pequenas frações de Milivolts."
    fig.text(0.5, 0.02, texto, ha="center", fontsize=10, style="italic", 
             bbox=dict(facecolor="yellow", alpha=0.2, boxstyle="round,pad=0.5"))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12) # Espaco para a caixa de texto
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  -> Grafico salvo em: {output_path}")


def plot_convergencia(R, L, C, V0, t_final, output_path="convergencia.png"):
    """
    Gráfico 3: Análise de convergência - como o erro diminui
    quando aumentamos o número de passos.
    """
    # Testar com diferentes quantidades de passos
    lista_n_steps = [10, 20, 50, 100, 200, 500, 1000]
    erros_max_euler = []
    erros_max_rk4 = []

    for n in lista_n_steps:
        # Calcular solução analítica (referência)
        q_analitico, _ = analytic_solution(R, L, C, V0, t_final, n)

        # Calcular soluções numéricas
        q_euler, q_rk4, _ = simulate_rlc(R, L, C, V0, t_final, n)

        # Calcular erro máximo de cada método
        _, erro_euler = calcular_erros(q_euler, q_analitico)
        _, erro_rk4 = calcular_erros(q_rk4, q_analitico)

        erros_max_euler.append(erro_euler / C)  # converter para tensão
        erros_max_rk4.append(erro_rk4 / C)

    plt.figure(figsize=(11, 6))

    plt.loglog(lista_n_steps, erros_max_euler, "o-", color="red",
               label="Euler (Taxa Lenta)", linewidth=2, markersize=8)
    plt.loglog(lista_n_steps, erros_max_rk4, "s-", color="green",
               label="RK4 (Taxa Rápida)", linewidth=2, markersize=8)

    # Textos explicativos para facilitar compreensao
    plt.text(12, erros_max_euler[0]*0.5, "Erro cai devagar\n(O(h) - Linear)", 
             color="darkred", fontsize=11, fontweight="bold", 
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
    plt.text(12, erros_max_rk4[0]*0.05, "Erro despenca rapidamente\n(O(h^4) - 4ª Ordem)", 
             color="darkgreen", fontsize=11, fontweight="bold",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    plt.xlabel("Número de passos da simulação (n)", fontsize=12, fontweight="bold")
    plt.ylabel("Erro máximo acumulado na tensão (V)", fontsize=12, fontweight="bold")
    plt.title("Análise de Convergência: Como o erro cai ao investir em mais passos", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.4, which="both", linestyle=":")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  -> Grafico salvo em: {output_path}")


# Manter compatibilidade com a função antiga
def plot_results(R, L, C, V0, t_final, n_steps, output_path="comparison.png"):
    """Função de compatibilidade - gera o gráfico original."""
    q_analitico, t = analytic_solution(R, L, C, V0, t_final, n_steps)
    q_euler, q_rk4, _ = simulate_rlc(R, L, C, V0, t_final, n_steps)

    v_analitico = [q / C for q in q_analitico]
    v_euler = [q / C for q in q_euler]
    v_rk4 = [q / C for q in q_rk4]

    plot_comparacao(t, v_analitico, v_euler, v_rk4, output_path)


# =====================================================================
# PARTE 5 - FUNÇÃO PRINCIPAL
# =====================================================================

def main():
    """Funcao principal que executa toda a simulacao."""

    # -------------------------------------------------------
    # PASSO 1: Definir os parâmetros do circuito RLC
    # -------------------------------------------------------
    # Valores típicos para um sensor IoT com supercapacitor
    R = 2.0        # Resistência total do circuito (Ohms)
    L = 0.01       # Indutância parasita das trilhas (Henrys) = 10 mH
    C = 1e-4       # Capacitância do supercapacitor (Farads) = 100 uF
    V0 = 3.3       # Tensão inicial de operação (Volts)
    t_final = 0.015 # Tempo final da simulação (segundos) = 15 ms
    n_steps = 60   # Número de passos de tempo

    print("=" * 65)
    print("  SIMULACAO NUMERICA - CIRCUITO RLC SERIE")
    print("  Descarga de Capacitor em Sensor IoT")
    print("=" * 65)

    # -------------------------------------------------------
    # PASSO 2: Mostrar os parâmetros escolhidos
    # -------------------------------------------------------
    print("\n--- Parametros do Circuito ---")
    print(f"  R  = {R} Ohms (resistencia)")
    print(f"  L  = {L} H (indutancia) = {L*1000:.1f} mH")
    print(f"  C  = {C} F (capacitancia) = {C*1e6:.0f} uF")
    print(f"  V0 = {V0} V (tensao inicial)")
    print(f"  Tempo final = {t_final*1000:.1f} ms")
    print(f"  Passos = {n_steps}")
    print(f"  h (tamanho do passo) = {t_final/n_steps*1000:.4f} ms")

    # Calcular e mostrar o tipo de amortecimento
    alpha = R / (2.0 * L)
    omega0 = 1.0 / math.sqrt(L * C)
    print(f"\n  alpha (amortecimento) = {alpha:.2f}")
    print(f"  omega0 (freq. natural) = {omega0:.2f}")
    if alpha > omega0:
        print("  Regime: SUPERAMORTECIDO (sem oscilacao)")
    elif abs(alpha - omega0) < 1e-10:
        print("  Regime: CRITICAMENTE AMORTECIDO")
    else:
        print("  Regime: SUBAMORTECIDO (com oscilacao)")

    # -------------------------------------------------------
    # PASSO 3: Executar os cálculos
    # -------------------------------------------------------
    print("\n--- Executando Simulacoes ---")

    # Solucao analitica (referencia exata)
    q_analitico, t = analytic_solution(R, L, C, V0, t_final, n_steps)

    # Solucoes numericas (Euler e RK4)
    q_euler, q_rk4, _ = simulate_rlc(R, L, C, V0, t_final, n_steps)

    # Converter carga para tensao: V = q / C
    v_analitico = [q / C for q in q_analitico]
    v_euler = [q / C for q in q_euler]
    v_rk4 = [q / C for q in q_rk4]

    # -------------------------------------------------------
    # PASSO 4: Calcular os erros
    # -------------------------------------------------------
    erros_euler, erro_max_euler = calcular_erros(q_euler, q_analitico)
    erros_rk4, erro_max_rk4 = calcular_erros(q_rk4, q_analitico)

    print(f"  Erro maximo Euler: {erro_max_euler/C:.6e} V")
    print(f"  Erro maximo RK4:   {erro_max_rk4/C:.6e} V")

    # -------------------------------------------------------
    # PASSO 5: Mostrar tabela de resultados
    # -------------------------------------------------------
    print("\n--- Tabela de Resultados (pontos selecionados) ---")
    print("-" * 90)
    print(f"{'t (ms)':>8} | {'V_Analitico (V)':>15} | {'V_Euler (V)':>12} | "
          f"{'V_RK4 (V)':>10} | {'Erro Euler (V)':>14} | {'Erro RK4 (V)':>12}")
    print("-" * 90)

    # Selecionar pontos: inicio, 1/4, 1/2, 3/4, final
    indices = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]
    for idx in indices:
        t_ms = t[idx] * 1000  # converter para milissegundos
        err_euler = abs(v_euler[idx] - v_analitico[idx])
        err_rk4 = abs(v_rk4[idx] - v_analitico[idx])
        print(f"{t_ms:>8.3f} | {v_analitico[idx]:>15.6f} | {v_euler[idx]:>12.6f} | "
              f"{v_rk4[idx]:>10.6f} | {err_euler:>14.6e} | {err_rk4:>12.6e}")

    print("-" * 90)

    # -------------------------------------------------------
    # PASSO 6: Gerar os gráficos
    # -------------------------------------------------------
    print("\n--- Gerando Graficos ---")

    # Grafico 1: Comparacao dos metodos
    plot_comparacao(t, v_analitico, v_euler, v_rk4, "artigo/comparacao_metodos.png")

    # Grafico 2: Erro absoluto ao longo do tempo
    plot_erro_absoluto(t, erros_euler, erros_rk4, C, "artigo/erro_absoluto.png")

    # Grafico 3: Convergencia (erro vs numero de passos)
    plot_convergencia(R, L, C, V0, t_final, "artigo/convergencia.png")

    # -------------------------------------------------------
    # PASSO 7: Resumo final
    # -------------------------------------------------------
    print("\n--- Conclusao ---")
    razao = erro_max_euler / erro_max_rk4 if erro_max_rk4 > 0 else float("inf")
    print(f"  O RK4 foi {razao:.0f}x mais preciso que o Euler para {n_steps} passos.")
    print("  Simulacao concluida com sucesso!")
    print("=" * 65)


if __name__ == "__main__":
    main()
