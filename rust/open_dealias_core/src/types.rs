use ndarray::ArrayD;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DealiasError {
    #[error("nyquist must be positive, got {0}")]
    InvalidNyquist(f64),
    #[error("shift2d expects a 2D array, got {0}D")]
    Expected2D(usize),
    #[error("shift3d expects a 3D array, got {0}D")]
    Expected3D(usize),
    #[error("max_abs_fold must be non-negative, got {0}")]
    InvalidMaxAbsFold(i16),
    #[error("shape mismatch: {0}")]
    ShapeMismatch(&'static str),
}

pub type Result<T> = std::result::Result<T, DealiasError>;

pub struct DualPrfResult {
    pub velocity: ArrayD<f64>,
    pub folds: ArrayD<i16>,
    pub confidence: ArrayD<f64>,
    pub reference: ArrayD<f64>,
    pub low_valid_gates: usize,
    pub high_valid_gates: usize,
    pub paired_gates: usize,
    pub low_branch_mean_fold: f64,
    pub high_branch_mean_fold: f64,
    pub mean_pair_gap: f64,
    pub max_pair_gap: f64,
}

pub struct Es90RadialResult {
    pub velocity: ArrayD<f64>,
    pub folds: ArrayD<i16>,
    pub confidence: ArrayD<f64>,
    pub reference: ArrayD<f64>,
    pub seed_index: Option<usize>,
}

pub struct Es90SweepResult {
    pub velocity: ArrayD<f64>,
    pub folds: ArrayD<i16>,
    pub confidence: ArrayD<f64>,
    pub reference: ArrayD<f64>,
}

pub struct Zw06Result {
    pub velocity: ArrayD<f64>,
    pub folds: ArrayD<i16>,
    pub confidence: ArrayD<f64>,
    pub reference: ArrayD<f64>,
    pub seeded_gates: usize,
    pub assigned_gates: usize,
    pub iterations_used: usize,
}

pub struct VariationalResult {
    pub velocity: ArrayD<f64>,
    pub folds: ArrayD<i16>,
    pub confidence: ArrayD<f64>,
    pub iterations_used: usize,
    pub changed_gates: usize,
}

pub struct RegionGraphResult {
    pub velocity: ArrayD<f64>,
    pub folds: ArrayD<i16>,
    pub confidence: ArrayD<f64>,
    pub reference: ArrayD<f64>,
    pub region_count: usize,
    pub assigned_regions: usize,
    pub seed_region: Option<usize>,
    pub block_shape: (usize, usize),
    pub merge_iterations: usize,
    pub wrap_azimuth: bool,
    pub average_fold: f64,
    pub regions_with_reference: usize,
    pub block_grid_shape: (usize, usize),
}

pub struct RecursiveResult {
    pub velocity: ArrayD<f64>,
    pub folds: ArrayD<i16>,
    pub confidence: ArrayD<f64>,
    pub reference: ArrayD<f64>,
    pub leaf_count: usize,
    pub max_depth: usize,
    pub split_texture_fraction: f64,
    pub reference_weight: f64,
    pub wrap_azimuth: bool,
    pub root_texture: f64,
    pub bootstrap_method: &'static str,
    pub bootstrap_region_count: usize,
    pub method: &'static str,
}

pub struct Volume3DResult {
    pub velocity: ArrayD<f64>,
    pub folds: ArrayD<i16>,
    pub confidence: ArrayD<f64>,
    pub reference: ArrayD<f64>,
    pub seed_sweep: usize,
    pub iterations_used: usize,
    pub sweep_order: Vec<usize>,
    pub per_sweep_valid_gates: Vec<usize>,
    pub per_sweep_seeded_gates: Vec<usize>,
    pub per_sweep_assigned_gates: Vec<usize>,
    pub per_sweep_iterations_used: Vec<usize>,
}

pub struct VadFitResult {
    pub u: f64,
    pub v: f64,
    pub offset: f64,
    pub rms: f64,
    pub iterations: usize,
    pub reference: ArrayD<f64>,
}

pub struct Xu11Result {
    pub velocity: ArrayD<f64>,
    pub folds: ArrayD<i16>,
    pub confidence: ArrayD<f64>,
    pub reference: ArrayD<f64>,
    pub u: f64,
    pub v: f64,
    pub offset: f64,
    pub vad_rms: f64,
    pub vad_iterations: usize,
    pub method: &'static str,
}

pub struct Jh01SweepResult {
    pub velocity: ArrayD<f64>,
    pub folds: ArrayD<i16>,
    pub confidence: ArrayD<f64>,
    pub reference: ArrayD<f64>,
    pub method: &'static str,
    pub valid_gates: usize,
    pub assigned_gates: usize,
    pub unresolved_gates: usize,
    pub resolved_fraction: f64,
}

pub struct Jh01VolumeResult {
    pub velocity: ArrayD<f64>,
    pub folds: ArrayD<i16>,
    pub confidence: ArrayD<f64>,
    pub reference: ArrayD<f64>,
    pub elevation_order_desc: Vec<usize>,
    pub per_sweep_valid_gates: Vec<usize>,
    pub per_sweep_assigned_gates: Vec<usize>,
    pub per_sweep_unresolved_gates: Vec<usize>,
    pub per_sweep_resolved_fraction: Vec<f64>,
    pub valid_gates: usize,
    pub assigned_gates: usize,
    pub unresolved_gates: usize,
    pub resolved_fraction: f64,
}

#[derive(Clone)]
pub struct MlModelState {
    pub weights: Vec<f64>,
    pub feature_names: Vec<String>,
    pub ridge: f64,
    pub train_rmse: f64,
    pub mode: String,
    pub nyquist: Option<f64>,
}

pub struct MlDealiasResult {
    pub velocity: ArrayD<f64>,
    pub folds: ArrayD<i16>,
    pub confidence: ArrayD<f64>,
    pub reference: ArrayD<f64>,
    pub trained_from: String,
    pub train_rmse: f64,
    pub ridge: f64,
    pub feature_names: Vec<String>,
    pub refine_method: Option<String>,
    pub refine_iterations: Option<usize>,
}
