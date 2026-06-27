# =============================================================================
# Módulo: Análise de Erros
# Descrição: Classe para cálculo e análise de erros numéricos
# =============================================================================

from typing import List, Tuple


class AnalisadorErros:
    """
    Calcula e analisa erros entre soluções numéricas e analíticas.
    """
    
    @staticmethod
    def calcular_erro_absoluto(
        valores_numericos: List[float],
        valores_analiticos: List[float]
    ) -> Tuple[List[float], float]:
        """
        Calcula o erro absoluto em cada ponto.
        
        Parâmetros:
            valores_numericos: Valores calculados pelo método numérico
            valores_analiticos: Valores da solução analítica (referência)
        
        Retorna:
            Tupla (erros, erro_max) com lista de erros e o erro máximo
        """
        if len(valores_numericos) != len(valores_analiticos):
            raise ValueError("Os vetores devem ter o mesmo tamanho.")
        
        erros = []
        for i in range(len(valores_numericos)):
            erro = abs(valores_numericos[i] - valores_analiticos[i])
            erros.append(erro)
        
        erro_max = max(erros) if erros else 0.0
        return erros, erro_max
    
    @staticmethod
    def calcular_erro_relativo(
        valores_numericos: List[float],
        valores_analiticos: List[float]
    ) -> Tuple[List[float], float]:
        """
        Calcula o erro relativo em cada ponto.
        
        Evita divisão por zero quando a solução analítica é zero.
        
        Retorna:
            Tupla (erros_rel, erro_rel_max) com lista de erros relativos e máximo
        """
        if len(valores_numericos) != len(valores_analiticos):
            raise ValueError("Os vetores devem ter o mesmo tamanho.")
        
        erros_rel = []
        eps = 1e-16  # Pequeno valor para evitar divisão por zero
        
        for i in range(len(valores_numericos)):
            denom = abs(valores_analiticos[i]) + eps
            erro_rel = abs(valores_numericos[i] - valores_analiticos[i]) / denom
            erros_rel.append(erro_rel)
        
        erro_rel_max = max(erros_rel) if erros_rel else 0.0
        return erros_rel, erro_rel_max
    
    @staticmethod
    def calcular_norma_l2(
        valores_numericos: List[float],
        valores_analiticos: List[float]
    ) -> float:
        """
        Calcula a norma L2 (raiz quadrada da soma dos quadrados dos erros).
        """
        if len(valores_numericos) != len(valores_analiticos):
            raise ValueError("Os vetores devem ter o mesmo tamanho.")
        
        soma_quadrados = 0.0
        for i in range(len(valores_numericos)):
            erro = valores_numericos[i] - valores_analiticos[i]
            soma_quadrados += erro ** 2
        
        n = len(valores_numericos)
        return (soma_quadrados / n) ** 0.5 if n > 0 else 0.0
    
    @staticmethod
    def resumo_erros(
        valores_numericos: List[float],
        valores_analiticos: List[float],
        nome_metodo: str
    ) -> dict:
        """
        Calcula um resumo completo dos erros.
        
        Retorna um dicionário com:
            - erro_max
            - erro_rel_max
            - erro_l2
            - nome_metodo
        """
        erros_abs, erro_max = AnalisadorErros.calcular_erro_absoluto(
            valores_numericos, valores_analiticos
        )
        erros_rel, erro_rel_max = AnalisadorErros.calcular_erro_relativo(
            valores_numericos, valores_analiticos
        )
        erro_l2 = AnalisadorErros.calcular_norma_l2(
            valores_numericos, valores_analiticos
        )
        
        return {
            "nome_metodo": nome_metodo,
            "erro_max": erro_max,
            "erro_rel_max": erro_rel_max,
            "erro_l2": erro_l2,
            "erros_abs": erros_abs,
            "erros_rel": erros_rel
        }
