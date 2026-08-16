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

# Algorithm version per method, folded into the ledger unit hash. BUMP THIS whenever the
# numerical definition of a method changes, so that `ledger.has_unit` cannot silently serve
# rows computed by an older algorithm. This is not cosmetic: the v1 softmode grid was produced
# by a denominator-6 q-search on 2x2x2 force constants, that search was replaced by the
# FC-commensurate one in e592e86, and because `settings` carried only the supercell the hash
# was unchanged -- so every stale row was skipped as "already present" instead of recomputed.
METHOD_VERSION = {
    "harmonic": 1,
    "softmode": 3,    # v3 = screen EVERY imaginary commensurate mode; unstable if ANY condenses
                      # v2 = FC-commensurate q-search + acoustic mask by |omega| (was: 3 lowest)
    "sscha": 2,       # v2 = acoustic modes identified by |omega|, not by sort order
    "hiphive": 1,
    "rattled": 1,
    "tdep": 1,
    "md_distort": 1,
}

SUPPORTED_MODELS = (
    "mace_mp0",
    "chgnet",
    "orb_v2",
    "orb_v3",
    "sevennet0",
    "mattersim",
)
