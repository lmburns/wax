#![cfg(feature = "diagnostics-metadata")]

use miette::SourceSpan;

use crate::token::{Annotation, Token};

#[derive(Clone, Debug)]
pub struct CapturingToken {
    index: usize,
    span: SourceSpan,
}

impl CapturingToken {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn span(&self) -> SourceSpan {
        self.span.clone()
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
