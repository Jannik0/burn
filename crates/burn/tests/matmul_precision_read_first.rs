//! Setting the f32 matmul precision after it was read. See `matmul_precision.rs` for how to run.
#![cfg(feature = "cubek")]

use burn::{
    Float32MatmulPrecision, Float32MatmulPrecisionError, float32_matmul_precision,
    set_float32_matmul_precision,
};

#[test]
fn set_after_first_read_is_refused() {
    // No configuration file sets `allow_tf32` here, so reading fixes the default.
    assert_eq!(float32_matmul_precision(), Float32MatmulPrecision::High);

    assert_eq!(
        set_float32_matmul_precision(Float32MatmulPrecision::Highest),
        Err(Float32MatmulPrecisionError {
            current: Float32MatmulPrecision::High
        })
    );
    assert_eq!(float32_matmul_precision(), Float32MatmulPrecision::High);
}
