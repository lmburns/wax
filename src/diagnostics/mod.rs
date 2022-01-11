#![cfg(any(feature = "diagnostics-inspect", feature = "diagnostics-report"))]

pub(crate) mod inspect;
pub(crate) mod report;

#[cfg_attr(docsrs, doc(cfg(feature = "diagnostics-inspect")))]
pub(crate) type Span = (usize, usize);
