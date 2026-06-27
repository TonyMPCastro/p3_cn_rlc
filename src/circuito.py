# =============================================================================
# Módulo: Circuito RLC
# Descrição: Classe que encapsula os parâmetros e propriedades do circuito RLC
# =============================================================================

import math
from dataclasses import dataclass


@dataclass
class CircuitoRLC:
    """
    Representa um circuito RLC série em descargas capacitivas.
    
    Atributos:
        R: Resistência em Ohms
        L: Indutância em Henrys
        C: Capacitância em Farads
        V0: Tensão inicial no capacitor em Volts
    """
    
    R: float  # Resistência (Ohms)
    L: float  # Indutância (Henrys)
    C: float  # Capacitância (Farads)
    V0: float  # Tensão inicial (Volts)
    
    def __post_init__(self):
        """Valida os parâmetros após inicialização."""
        if self.R < 0:
            raise ValueError("Resistência R deve ser não-negativa.")
        if self.L <= 0:
            raise ValueError("Indutância L deve ser positiva.")
        if self.C <= 0:
            raise ValueError("Capacitância C deve ser positiva.")
        if self.V0 < 0:
            raise ValueError("Tensão V0 deve ser não-negativa.")
    
    @property
    def q0(self) -> float:
        """Carga inicial no capacitor (Coulombs)."""
        return self.C * self.V0
    
    @property
    def alpha(self) -> float:
        """Coeficiente de amortecimento."""
        return self.R / (2.0 * self.L)
    
    @property
    def omega0(self) -> float:
        """Frequência natural não amortecida (rad/s)."""
        return 1.0 / math.sqrt(self.L * self.C)
    
    @property
    def discriminante(self) -> float:
        """Discriminante para classificação do regime."""
        return self.alpha**2 - self.omega0**2
    
    @property
    def regime(self) -> str:
        """Retorna o tipo de regime: 'superamortecido', 'criticamente_amortecido' ou 'subamortecido'."""
        delta = self.discriminante
        if abs(delta) < 1e-14:
            return "criticamente_amortecido"
        elif delta > 0:
            return "superamortecido"
        else:
            return "subamortecido"
    
    def __str__(self) -> str:
        """Representação em string dos parâmetros do circuito."""
        return (
            f"CircuitoRLC(R={self.R} Ω, L={self.L*1000:.1f} mH, "
            f"C={self.C*1e6:.0f} µF, V0={self.V0} V)\n"
            f"  Regime: {self.regime.upper().replace('_', ' ')}\n"
            f"  α = {self.alpha:.2f}, ω₀ = {self.omega0:.2f}"
        )
