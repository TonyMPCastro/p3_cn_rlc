import unittest

from rlc_simulation import (
    analytic_solution,
    calcular_erros,
    laplace_transform_solution,
    simulate_rlc,
)


class RlcSimulationTests(unittest.TestCase):
    """Testes para validar a simulação do circuito RLC."""

    # Parâmetros padrão usados nos testes
    R = 10.0
    L = 0.01
    C = 1e-4
    V0 = 3.3
    t_final = 0.01

    # -------------------------------------------------------
    # Testes de condição inicial
    # -------------------------------------------------------

    def test_analytic_solution_starts_at_initial_charge(self):
        """A solução analítica deve começar com q(0) = C * V0."""
        q, t = analytic_solution(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=10
        )
        self.assertEqual(len(q), len(t))
        self.assertAlmostEqual(q[0], self.C * self.V0, places=12)

    def test_simulation_returns_expected_length_and_initial_value(self):
        """Euler e RK4 devem retornar n_steps+1 pontos e começar em q0."""
        q_euler, q_rk4, t = simulate_rlc(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=10
        )
        self.assertEqual(len(q_euler), 11)
        self.assertEqual(len(q_rk4), 11)
        self.assertEqual(len(t), 11)
        self.assertAlmostEqual(q_euler[0], self.C * self.V0, places=12)
        self.assertAlmostEqual(q_rk4[0], self.C * self.V0, places=12)

    def test_laplace_transform_solution_matches_initial_condition(self):
        """A solução via Laplace deve começar com q(0) = C * V0."""
        q, t = laplace_transform_solution(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=10
        )
        self.assertEqual(len(q), len(t))
        self.assertAlmostEqual(q[0], self.C * self.V0, places=12)

    # -------------------------------------------------------
    # Testes de convergência
    # -------------------------------------------------------

    def test_euler_converge_ao_aumentar_passos(self):
        """O erro do Euler deve diminuir quando aumentamos o número de passos."""
        q_analitico_50, _ = analytic_solution(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=50
        )
        q_euler_50, _, _ = simulate_rlc(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=50
        )
        _, erro_50 = calcular_erros(q_euler_50, q_analitico_50)

        q_analitico_200, _ = analytic_solution(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=200
        )
        q_euler_200, _, _ = simulate_rlc(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=200
        )
        _, erro_200 = calcular_erros(q_euler_200, q_analitico_200)

        # Com mais passos, o erro deve ser menor
        self.assertLess(erro_200, erro_50)

    def test_rk4_converge_ao_aumentar_passos(self):
        """O erro do RK4 deve diminuir quando aumentamos o número de passos."""
        q_analitico_20, _ = analytic_solution(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=20
        )
        _, q_rk4_20, _ = simulate_rlc(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=20
        )
        _, erro_20 = calcular_erros(q_rk4_20, q_analitico_20)

        q_analitico_100, _ = analytic_solution(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=100
        )
        _, q_rk4_100, _ = simulate_rlc(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=100
        )
        _, erro_100 = calcular_erros(q_rk4_100, q_analitico_100)

        # Com mais passos, o erro deve ser menor
        self.assertLess(erro_100, erro_20)

    def test_rk4_mais_preciso_que_euler(self):
        """Para o mesmo número de passos, o RK4 deve ter erro menor que o Euler."""
        n = 100
        q_analitico, _ = analytic_solution(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=n
        )
        q_euler, q_rk4, _ = simulate_rlc(
            R=self.R, L=self.L, C=self.C, V0=self.V0,
            t_final=self.t_final, n_steps=n
        )

        _, erro_euler = calcular_erros(q_euler, q_analitico)
        _, erro_rk4 = calcular_erros(q_rk4, q_analitico)

        # RK4 deve ser mais preciso (erro menor)
        self.assertLess(erro_rk4, erro_euler)

    # -------------------------------------------------------
    # Testes de validação de parâmetros
    # -------------------------------------------------------

    def test_rejeita_L_negativo(self):
        """Deve levantar erro se L for negativo."""
        with self.assertRaises(ValueError):
            simulate_rlc(R=10, L=-0.01, C=1e-4, V0=3.3, t_final=0.01, n_steps=10)

    def test_rejeita_n_steps_zero(self):
        """Deve levantar erro se n_steps for zero."""
        with self.assertRaises(ValueError):
            simulate_rlc(R=10, L=0.01, C=1e-4, V0=3.3, t_final=0.01, n_steps=0)


if __name__ == "__main__":
    unittest.main()
