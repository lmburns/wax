#![cfg(any(feature = "diagnostics-inspect", feature = "diagnostics-report"))]

pub mod inspect;
pub mod report;

pub type Span = (usize, usize);
