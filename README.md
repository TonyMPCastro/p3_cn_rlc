
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

Para executar a simulação e gerar os gráficos comparativos:

```bash
python rlc_simulation.py
```

O script gera os seguintes arquivos:

- `comparacao_metodos.png` — Curvas de tensão V(t) dos 3 métodos sobrepostas
- `erro_absoluto.png` — Erro absoluto de Euler e RK4 ao longo do tempo
- `convergencia.png` — Análise de convergência (erro máximo vs. número de passos)

Além dos gráficos, uma tabela de resultados é impressa no terminal com valores em pontos selecionados.

## Estrutura do projeto

- `rlc_simulation.py`: implementação dos métodos numéricos, solução analítica e geração de gráficos.
- `tests/test_simulation.py`: testes para validar a implementação (condições iniciais, convergência, precisão).
- `main.tex`: artigo científico em LaTeX com a análise completa.
- `referencias.bib`: referências bibliográficas no formato BibTeX.

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

## Testes

Para executar os testes:

```bash
python -m unittest discover -s tests -v
```

Os testes verificam:
- Condições iniciais corretas
- Convergência do Euler e RK4 ao aumentar o número de passos
- RK4 é mais preciso que Euler para o mesmo passo
- Validação de parâmetros inválidos
