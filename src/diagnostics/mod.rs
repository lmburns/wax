#![cfg(any(feature = "diagnostics-error", feature = "diagnostics-inspect"))]

pub mod error;
pub mod inspect;

pub type Span = (usize, usize);
