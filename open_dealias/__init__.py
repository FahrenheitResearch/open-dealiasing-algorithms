"""open_dealias: compact public radar velocity dealiasing algorithms.

This package intentionally implements public, paper-based algorithm families rather
than guessing at proprietary internals.
"""

from ._core import fold_counts, shift2d, shift3d, unfold_to_reference, wrap_to_nyquist
from .continuity import dealias_radial_es90, dealias_sweep_es90
from .dual_prf import dealias_dual_prf
from .fourdd import dealias_sweep_jh01, dealias_volume_jh01
from .ml import LinearBranchModel, dealias_sweep_ml, fit_ml_reference_model
from .multipass import DEFAULT_PASSES, dealias_sweep_zw06
from .nexrad import (
    NEXRAD_LEVEL2_BUCKET,
    build_nexrad_prefix,
    download_nexrad_key,
    find_nexrad_key,
    list_nexrad_keys,
    load_nexrad_sweep,
    parse_nexrad_key,
    select_nexrad_key,
)
from .palette import PalColorTable, PaletteStop, load_pal_table
from .qc import apply_velocity_qc, build_velocity_qc_mask, estimate_velocity_texture
from .result_state import ResultProvenance, ResultState, ResultStatus, attach_result_state
from .recursive import dealias_sweep_recursive
from .region_graph import dealias_sweep_region_graph
from .types import DealiasResult, RadarSweep, VadFit
from .vad import build_reference_from_uv, dealias_sweep_xu11, estimate_uniform_wind_vad
from .variational import dealias_sweep_variational
from .volume3d import dealias_volume_3d

__all__ = [
    'DealiasResult',
    'LinearBranchModel',
    'RadarSweep',
    'VadFit',
    'wrap_to_nyquist',
    'unfold_to_reference',
    'fold_counts',
    'shift2d',
    'shift3d',
    'NEXRAD_LEVEL2_BUCKET',
    'build_nexrad_prefix',
    'parse_nexrad_key',
    'select_nexrad_key',
    'list_nexrad_keys',
    'find_nexrad_key',
    'download_nexrad_key',
    'load_nexrad_sweep',
    'PaletteStop',
    'PalColorTable',
    'load_pal_table',
    'ResultStatus',
    'ResultProvenance',
    'ResultState',
    'attach_result_state',
    'estimate_velocity_texture',
    'build_velocity_qc_mask',
    'apply_velocity_qc',
    'dealias_radial_es90',
    'dealias_sweep_es90',
    'dealias_sweep_zw06',
    'estimate_uniform_wind_vad',
    'build_reference_from_uv',
    'dealias_sweep_xu11',
    'dealias_sweep_jh01',
    'dealias_volume_jh01',
    'dealias_sweep_region_graph',
    'dealias_sweep_recursive',
    'dealias_dual_prf',
    'dealias_volume_3d',
    'dealias_sweep_variational',
    'fit_ml_reference_model',
    'dealias_sweep_ml',
    'DEFAULT_PASSES',
]
