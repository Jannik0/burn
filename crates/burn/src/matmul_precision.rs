//! Process-wide precision of `f32` matrix multiplication.

use core::fmt;

/// How precisely `f32` matrix multiplication is computed, named after PyTorch's
/// `torch.set_float32_matmul_precision` levels.
///
/// Only CubeCL backends choose between these levels; other backends always compute `f32`
/// matmul in `f32`. Convolution is not affected and may use TF32 at either level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Float32MatmulPrecision {
    /// Keep `f32` operands in `f32`. Accelerated kernels that need narrower operands are not
    /// selected, which may be slower.
    Highest,
    /// Allow `f32` operands to be rounded to TF32 (10 mantissa bits) on devices that
    /// accelerate it, such as NVIDIA GPUs from Ampere on. This is the default.
    High,
}

#[cfg(feature = "cubek")]
impl Float32MatmulPrecision {
    fn allows_tf32(self) -> bool {
        match self {
            Self::Highest => false,
            Self::High => true,
        }
    }

    fn from_allow_tf32(allow_tf32: bool) -> Self {
        match allow_tf32 {
            true => Self::High,
            false => Self::Highest,
        }
    }
}

/// Returned by [`set_float32_matmul_precision`] when the precision was already fixed to
/// another level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Float32MatmulPrecisionError {
    /// The precision that stays in effect.
    pub current: Float32MatmulPrecision,
}

impl fmt::Display for Float32MatmulPrecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the f32 matmul precision is already fixed to {:?}; set it before the first matmul",
            self.current
        )
    }
}

impl core::error::Error for Float32MatmulPrecisionError {}

/// Sets the precision of `f32` matrix multiplication for the whole process.
///
/// The precision is read once, when a kernel is first selected or the configuration is first
/// read, and is fixed afterwards. Call this at startup, before any tensor operation. Calling it
/// again with the level in effect succeeds.
///
/// The level replaces `[cubek.matmul] allow_tf32` from `burn.toml` or `cubek.toml`; other
/// settings from those files are kept. It applies to every device, with or without fusion and
/// autotuning. A remote backend uses the precision of the process that executes the kernels.
///
/// Without a CubeCL backend this does nothing and returns `Ok`.
///
/// # Errors
///
/// Returns [`Float32MatmulPrecisionError`] if the precision was already fixed to another level.
///
/// # Example
///
/// ```rust
/// use burn::{Float32MatmulPrecision, set_float32_matmul_precision};
///
/// set_float32_matmul_precision(Float32MatmulPrecision::Highest)
///     .expect("called before the first matmul");
/// ```
pub fn set_float32_matmul_precision(
    precision: Float32MatmulPrecision,
) -> Result<(), Float32MatmulPrecisionError> {
    #[cfg(feature = "cubek")]
    {
        use cubek::config::{CubekConfig, RuntimeConfig};

        let mut config = file_config();
        config.matmul.allow_tf32 = precision.allows_tf32();
        if CubekConfig::try_set(config) {
            return Ok(());
        }
        let current = float32_matmul_precision();
        if current != precision {
            return Err(Float32MatmulPrecisionError { current });
        }
    }
    #[cfg(not(feature = "cubek"))]
    let _ = precision;
    Ok(())
}

/// Returns the precision of `f32` matrix multiplication in effect.
///
/// With a CubeCL backend, this reads the configuration and so fixes the precision: a later
/// [`set_float32_matmul_precision`] to another level fails. Without one, this always returns
/// [`Float32MatmulPrecision::Highest`].
pub fn float32_matmul_precision() -> Float32MatmulPrecision {
    #[cfg(feature = "cubek")]
    {
        use cubek::config::{CubekConfig, RuntimeConfig};
        Float32MatmulPrecision::from_allow_tf32(CubekConfig::get().matmul.allow_tf32)
    }
    #[cfg(not(feature = "cubek"))]
    Float32MatmulPrecision::Highest
}

/// The configuration CubeK would load on its own, without fixing it.
#[cfg(feature = "cubek")]
fn file_config() -> cubek::config::CubekConfig {
    #[cfg(all(
        feature = "std",
        any(
            target_os = "windows",
            target_os = "linux",
            target_os = "macos",
            target_os = "android"
        )
    ))]
    {
        use cubek::config::{CubekConfig, RuntimeConfig};
        CubekConfig::from_current_dir().override_from_env()
    }
    #[cfg(not(all(
        feature = "std",
        any(
            target_os = "windows",
            target_os = "linux",
            target_os = "macos",
            target_os = "android"
        )
    )))]
    {
        Default::default()
    }
}
