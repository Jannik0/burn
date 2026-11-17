//! Setting the f32 matmul precision at startup.
//!
//! The precision is fixed once per process, so each order of calls is its own test binary. These
//! need a CubeCL backend; without one the setting does nothing:
//!
//! ```sh
//! cargo test -p burn --features cuda --test matmul_precision --test matmul_precision_read_first
//! ```
#![cfg(feature = "cubek")]

use burn::{
    Float32MatmulPrecision, Float32MatmulPrecisionError, float32_matmul_precision,
    set_float32_matmul_precision,
};

#[test]
fn set_before_first_read_fixes_the_level() {
    // `High` is the default, so only `Highest` shows that the call took effect.
    assert_eq!(
        set_float32_matmul_precision(Float32MatmulPrecision::Highest),
        Ok(())
    );
    assert_eq!(float32_matmul_precision(), Float32MatmulPrecision::Highest);

    // Repeating the level in effect is accepted; another level is refused and leaves it.
    assert_eq!(
        set_float32_matmul_precision(Float32MatmulPrecision::Highest),
        Ok(())
    );
    assert_eq!(
        set_float32_matmul_precision(Float32MatmulPrecision::High),
        Err(Float32MatmulPrecisionError {
            current: Float32MatmulPrecision::Highest
        })
    );
    assert_eq!(float32_matmul_precision(), Float32MatmulPrecision::Highest);
}
