
## Análise Numérica de Circuitos RLC Série: Modelagem da Descarga de Capacitores em Sensores IoT

**Autores:**
- Antonio Marcos Patrício Castro
- Felipe Carneiro de Albuquerque Sarmanho
- Pedro Kaic Ribeiro Veloso

**Instituição:** Centro de Ciências Exatas e Tecnologia (CCET)  
Universidade Federal do Maranhão (UFMA)

**Docente:** Prof. Me. Lucas Reis Abreu

---

## Descrição do Projeto

Este projeto implementa uma simulação numérica da descarga de um capacitor em um circuito RLC série, com foco em aplicações de backup de energia para sensores IoT. A solução é comparada entre:

- solução analítica obtida via Transformada de Laplace;
- método de Euler explícito;
- método de Runge-Kutta de quarta ordem (RK4).

## Requisitos

- Python 3.9+
- matplotlib
- numpy

## Instalação

No diretório do projeto, instale as dependências com:

```bash
pip install matplotlib numpy
```

# Simulação Numérica de um Circuito RLC Série

## Execução

### Simulação Completa

Para executar a simulação e gerar os gráficos comparativos:

```bash
python rlc_simulation.py
```

O script gera os seguintes arquivos na pasta `artigo/`:

- `comparacao_metodos.png` — Curvas de tensão V(t) dos 3 métodos sobrepostas (com zoom)
- `erro_absoluto.png` — Erro absoluto de Euler e RK4 ao longo do tempo (escala logarítmica)
- `convergencia.png` — Análise de convergência (erro máximo vs. número de passos)

### Exemplos de Uso Individual dos Módulos

Para ver exemplos de como usar os módulos isoladamente:

```bash
python src/exemplos_uso.py
```

Exemplos incluem:
- Criação de circuitos com diferentes regimes de amortecimento
- Comparação individual de métodos numéricos
- Análise de erros com múltiplas métricas
- Estudo de convergência parametrizado
- Utilização de classes individuais

## Estrutura do Projeto

```
c:\Projetos\p3_cn_rlc\
├── rlc_simulation.py          # Script principal (orquestrador)
├── src/                       # Pacote com módulos modularizados
│   ├── __init__.py
│   ├── circuito.py            # Classe CircuitoRLC
│   ├── solucao_analitica.py   # Classe SolucaoAnalitica
│   ├── metodos_numericos.py   # Classes MetodoEuler e MetodoRK4
│   ├── analise_erros.py       # Classe AnalisadorErros
│   ├── visualizacao.py        # Classe Visualizador
│   └── exemplos_uso.py        # 6 exemplos de uso
├── tests/
│   └── test_simulation.py     # Testes unitários
├── artigo/
│   ├── main.tex               # Artigo científico em LaTeX
│   ├── referencias.bib        # Referências bibliográficas
│   ├── comparacao_metodos.png # Gráfico 1: Comparação visual
│   ├── erro_absoluto.png      # Gráfico 2: Erro em tempo real
│   └── convergencia.png       # Gráfico 3: Ordem de convergência
├── ARQUITETURA.md             # Documentação detalhada da arquitetura
├── README.md                  # Este arquivo
└── .venv/                     # Ambiente virtual Python

## Como funciona

O circuito é modelado pela equação diferencial

$$
L\frac{d^2q(t)}{dt^2} + R\frac{dq(t)}{dt} + \frac{1}{C}q(t) = 0
$$

com condições iniciais

$$
q(0)=CV_0, \quad q'(0)=0.
$$

A solução analítica é derivada com base na Transformada de Laplace, partindo da equação diferencial do circuito e das condições iniciais. A simulação converte a EDO de segunda ordem em um sistema de primeira ordem e aplica os métodos numéricos manualmente, sem o uso de bibliotecas de integração prontas.

## Arquitetura Modular

O projeto foi refatorado em uma arquitetura orientada a objetos com separação clara de responsabilidades:

### Módulos Principais (em `src/`)

| Módulo | Classe | Responsabilidade |
|--------|--------|------------------|
| `circuito.py` | `CircuitoRLC` | Encapsula parâmetros (R, L, C, V0) e calcula propriedades (α, ω₀, regime) |
| `solucao_analitica.py` | `SolucaoAnalitica` | Calcula solução exata via Transformada de Laplace para 3 regimes |
| `metodos_numericos.py` | `MetodoEuler`, `MetodoRK4` | Integração numérica O(h) e O(h⁴) respectivamente |
| `analise_erros.py` | `AnalisadorErros` | Calcula erro absoluto, relativo e norma L2 |
| `visualizacao.py` | `Visualizador` | Gera 3 gráficos acadêmicos em alta resolução |

### Vantagens da Arquitetura

- ✅ **Modularidade**: Cada classe tem responsabilidade única
- ✅ **Reutilização**: Classes independentes e reutilizáveis
- ✅ **Testabilidade**: Fácil criar testes unitários
- ✅ **Extensibilidade**: Adicionar novos métodos é simples
- ✅ **Importabilidade**: Usar módulos em outros projetos

### Exemplo de Uso

```python
from src.circuito import CircuitoRLC
from src.metodos_numericos import MetodoEuler, MetodoRK4

# Criar circuito
circuito = CircuitoRLC(R=2.0, L=0.01, C=1e-4, V0=3.3)

# Usar métodos isoladamente
euler = MetodoEuler(circuito)
q_euler, i_euler, t = euler.simular(t_final=0.015, n_steps=100)

rk4 = MetodoRK4(circuito)
q_rk4, i_rk4, t = rk4.simular(t_final=0.015, n_steps=100)
```

## Testes

Para executar os testes unitários:

```bash
python -m pytest tests/ -v
# ou
python -m unittest discover -s tests -v
```

Os testes verificam:
- Condições iniciais corretas (q0, i0)
- Convergência do Euler e RK4 ao aumentar o número de passos
- RK4 é mais preciso que Euler para o mesmo passo
- Validação de parâmetros inválidos
- Cálculo correto de propriedades do circuito (α, ω₀)
