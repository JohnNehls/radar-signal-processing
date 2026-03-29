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
    """Electronic attack (EA) platform transmitter parameters.

    Attributes:
        txPower: Transmit power [W].
        txGain: Transmit antenna gain [linear].
        totalLosses: Total system losses [linear].
    """

    txPower: float
    txGain: float
    totalLosses: float


@dataclass
class SkinReturn:
    """Direct radar skin return from a target.

    Attributes:
        target: Target kinematics and RCS.
    """

    target: Target


@dataclass
class MemoryReturn:
    """DRFM memory-based electronic attack return.

    Attributes:
        target: Target kinematics (range/rangeRate) at the EA platform location.
        platform: EA transmitter parameters.  Required for physical amplitude
            mode; unused in SNR mode (which borrows ``target.rcs`` instead).
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

    target: Target
    platform: EaPlatform | None = None
    rdot_delta: float | None = None
    rdot_offset: float = 0.0
    range_offset: float = 0.0
    delay: float = 0.0
    vbm_noise_function: Callable | None = None
