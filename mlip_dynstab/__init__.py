"""mlip_dynstab — finite-temperature dynamic-stability stress test of foundation MLIPs.

The package is intentionally split so that heavy, mutually-incompatible MLIP backends are
imported lazily (one Python env per model). Structure/displacement/MD generation is
backend-agnostic; each model env only needs to supply an ASE calculator via
``calculators.get_calculator``.
"""

__version__ = "0.1.0"

# Stability threshold (THz). A mode below this is counted "imaginary"/unstable. The small
# negative tolerance absorbs finite-displacement / acoustic-sum-rule numerical noise near
# Gamma. Reported alongside every stability call; sensitivity is swept in analysis.
DEFAULT_IMAG_TOL_THZ = -0.1

SUPPORTED_MODELS = (
    "mace_mp0",
    "chgnet",
    "orb_v2",
    "orb_v3",
    "sevennet0",
    "mattersim",
)
