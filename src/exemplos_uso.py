# =============================================================================
# Exemplos de Uso dos Módulos
# Como usar as classes de forma independente
# =============================================================================

# ============ EXEMPLO 1: Usar apenas CircuitoRLC ===========
def exemplo_1_circuito():
    from src.circuito import CircuitoRLC
    
    # Criar instância
    circuito = CircuitoRLC(R=2.0, L=0.01, C=1e-4, V0=3.3)
    
    # Acessar propriedades
    print("Carga inicial:", circuito.q0, "Coulombs")
    print("Amortecimento:", circuito.alpha)
    print("Frequência natural:", circuito.omega0, "rad/s")
    print("Regime:", circuito.regime)


# ============ EXEMPLO 2: Comparar apenas 2 métodos ===========
def exemplo_2_metodos_numericos():
    from src.circuito import CircuitoRLC
    from src.metodos_numericos import MetodoEuler, MetodoRK4
    from src.solucao_analitica import SolucaoAnalitica
    
    circuito = CircuitoRLC(R=2.0, L=0.01, C=1e-4, V0=3.3)
    
    # Solução analítica como referência
    sol_ana = SolucaoAnalitica(circuito)
    q_ana, t = sol_ana.calcular(t_final=0.015, n_steps=100)
    
    # Métodos numéricos
    euler = MetodoEuler(circuito)
    rk4 = MetodoRK4(circuito)
    
    q_euler, i_euler, _ = euler.simular(t_final=0.015, n_steps=100)
    q_rk4, i_rk4, _ = rk4.simular(t_final=0.015, n_steps=100)
    
    print(f"Carga final (Analítico): {q_ana[-1]:.6e} C")
    print(f"Carga final (Euler):     {q_euler[-1]:.6e} C")
    print(f"Carga final (RK4):       {q_rk4[-1]:.6e} C")


# ============ EXEMPLO 3: Análise de erros ===========
def exemplo_3_analise_erros():
    from src.circuito import CircuitoRLC
    from src.metodos_numericos import MetodoEuler
    from src.solucao_analitica import SolucaoAnalitica
    from src.analise_erros import AnalisadorErros
    
    circuito = CircuitoRLC(R=2.0, L=0.01, C=1e-4, V0=3.3)
    
    sol_ana = SolucaoAnalitica(circuito)
    q_ana, t = sol_ana.calcular(0.015, 100)
    
    euler = MetodoEuler(circuito)
    q_euler, _, _ = euler.simular(0.015, 100)
    
    # Análise completa
    resumo = AnalisadorErros.resumo_erros(q_euler, q_ana, "Euler")
    
    print("Nome do método:", resumo['nome_metodo'])
    print("Erro máximo absoluto:", resumo['erro_max'])
    print("Erro máximo relativo:", resumo['erro_rel_max'])
    print("Norma L2:", resumo['erro_l2'])


# ============ EXEMPLO 5: Estudo de convergência customizado ===========
def exemplo_5_convergencia():
    from src.circuito import CircuitoRLC
    from src.metodos_numericos import MetodoEuler, MetodoRK4
    from src.solucao_analitica import SolucaoAnalitica
    from src.analise_erros import AnalisadorErros
    
    circuito = CircuitoRLC(R=2.0, L=0.01, C=1e-4, V0=3.3)
    
    # Testar diferentes resoluções
    resolutions = [10, 50, 100, 500, 1000]
    
    print("n_steps | Erro Euler | Erro RK4 | Razão")
    print("-" * 50)
    
    for n in resolutions:
        sol_ana = SolucaoAnalitica(circuito)
        q_ana, _ = sol_ana.calcular(0.015, n)
        
        euler = MetodoEuler(circuito)
        q_euler, _, _ = euler.simular(0.015, n)
        _, err_euler = AnalisadorErros.calcular_erro_absoluto(q_euler, q_ana)
        
        rk4 = MetodoRK4(circuito)
        q_rk4, _, _ = rk4.simular(0.015, n)
        _, err_rk4 = AnalisadorErros.calcular_erro_absoluto(q_rk4, q_ana)
        
        razao = err_euler / err_rk4 if err_rk4 > 0 else float('inf')
        print(f"{n:>7} | {err_euler:>10.2e} | {err_rk4:>8.2e} | {razao:>6.1f}x")


# ============ EXEMPLO 6: Solução customizada (regiões de amortecimento) ===========
def exemplo_6_circuitos_diferentes():
    from src.circuito import CircuitoRLC
    
    print("Testando diferentes regimes de amortecimento:\n")
    
    # 1. SUBAMORTECIDO (oscilações)
    c1 = CircuitoRLC(R=2.0, L=0.01, C=1e-4, V0=3.3)
    print(f"Circuito 1: R={c1.R}, Regime: {c1.regime}")
    
    # 2. CRITICAMENTE AMORTECIDO (transição)
    R_critico = 2 * (0.01 / 1e-4) ** 0.5
    c2 = CircuitoRLC(R=R_critico, L=0.01, C=1e-4, V0=3.3)
    print(f"Circuito 2: R={c2.R:.2f}, Regime: {c2.regime}")
    
    # 3. SUPERAMORTECIDO (sem oscilações)
    c3 = CircuitoRLC(R=10.0, L=0.01, C=1e-4, V0=3.3)
    print(f"Circuito 3: R={c3.R}, Regime: {c3.regime}")


if __name__ == "__main__":
    print("=" * 60)
    print("EXEMPLOS DE USO DOS MÓDULOS")
    print("=" * 60)
    
    print("\n[1] Propriedades do Circuito RLC")
    print("-" * 60)
    exemplo_1_circuito()
    
    print("\n[2] Comparação de Métodos Numéricos")
    print("-" * 60)
    exemplo_2_metodos_numericos()
    
    print("\n[3] Análise de Erros")
    print("-" * 60)
    exemplo_3_analise_erros()
    
    print("\n[5] Estudo de Convergência")
    print("-" * 60)
    exemplo_5_convergencia()
    
    print("\n[6] Diferentes Regimes de Amortecimento")
    print("-" * 60)
    exemplo_6_circuitos_diferentes()
    
    print("\n" + "=" * 60)
    print("Exemplos concluídos!")
    print("=" * 60)
