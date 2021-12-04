#![cfg(feature = "diagnostics-inspect")]

use crate::diagnostics::Span;
use crate::token::{Annotation, Token};

#[derive(Clone, Copy, Debug)]
pub struct CapturingToken {
    index: usize,
    span: Span,
}

impl CapturingToken {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

pub fn captures<'t, I>(tokens: I) -> impl 't + Clone + Iterator<Item = CapturingToken>
where
    I: IntoIterator<Item = &'t Token<'t, Annotation>>,
    I::IntoIter: 't + Clone,
{
    tokens
        .into_iter()
        .filter(|token| token.is_capturing())
        .enumerate()
        .map(|(index, token)| CapturingToken {
            index,
            span: token.annotation().clone(),
        })
}
