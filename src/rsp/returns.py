from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass
class Target:
    """Target kinematics and scattering parameters.

    Attributes:
        range: True target range [m].
        rangeRate: Radial velocity (positive = receding) [m/s].
        rcs: Radar cross section [m^2].  Required for skin return amplitude;
            unused by memory return amplitude in physical mode.
        sv: Steering vector component for a single array element [dimensionless].
            Defaults to 1 (isotropic / no array).
    """

    range: float
    rangeRate: float
    rcs: float | None = None
    sv: complex = 1


@dataclass
class EaPlatform:
    """Electronic attack (EA) platform — hardware and signal modulation parameters.

    All fields beyond the first three are modulation parameters that control how
    the stored pulse is modified before retransmission.

    Attributes:
        txPower: Transmit power [W].
        txGain: Transmit antenna gain [linear].
        totalLosses: Total system losses [linear].
        rdot_delta: VBM Doppler spread [m/s].  ``None`` disables VBM entirely.
        rdot_offset: Doppler frequency offset applied to the retransmitted
            pulse [m/s].  Defaults to 0.
        range_offset: Additional apparent range offset injected by the EA [m].
            Defaults to 0.
        delay: Additional time delay applied to the retransmitted pulse [s].
            Defaults to 0.
        vbm_noise_function: Phase-noise function used for VBM slow-time
            modulation.  ``None`` selects the default LFM noise function.
    """

    txPower: float
    txGain: float
    totalLosses: float
    rdot_delta: float | None = None
    rdot_offset: float = 0.0
    range_offset: float = 0.0
    delay: float = 0.0
    vbm_noise_function: Callable | None = None


@dataclass
class Return:
    """A simulated radar return — skin, memory-based EA, or both simultaneously.

    The two contributions are independent and additive:

    - Skin return fires when ``target.rcs is not None``.  The radar's own
      transmitted pulse reflects off the target.  ``target.rcs`` sets the
      physical amplitude via the two-way range equation.
    - Memory return fires when ``platform is not None``.  A DRFM jammer
      receives the pulse, stores it, and retransmits with the modulation
      specified in ``platform``.  Amplitude is derived from the one-way
      link equation.

    Both can be active on the same Return, modelling a jammer co-located
    with the target (set ``target.rcs`` and supply a ``platform``).

    Attributes:
        target: Target kinematics and (for skin) RCS.
        platform: EA platform parameters.  ``None`` disables memory return.
    """

    target: Target
    platform: EaPlatform | None = None
