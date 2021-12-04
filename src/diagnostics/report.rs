#![cfg(feature = "diagnostics-report")]

use itertools::Itertools as _;
use miette::{Diagnostic, LabeledSpan, SourceSpan};
use std::borrow::Cow;
use std::cmp;
use thiserror::Error;

use crate::token::{self, Annotation, TokenKind, Tokenized};

pub trait SourceSpanExt {
    fn union(&self, other: &SourceSpan) -> SourceSpan;
}

impl SourceSpanExt for SourceSpan {
    fn union(&self, other: &SourceSpan) -> SourceSpan {
        let start = cmp::min(self.offset(), other.offset());
        let end = cmp::max(self.offset() + self.len(), other.offset() + other.len());
        (start, end - start).into()
    }
}

#[derive(Clone, Debug)]
pub struct CompositeSourceSpan {
    label: Option<&'static str>,
    kind: CompositeKind,
}

impl CompositeSourceSpan {
    pub fn span(label: Option<&'static str>, span: SourceSpan) -> Self {
        CompositeSourceSpan {
            label,
            kind: CompositeKind::Span(span),
        }
    }

    pub fn correlated(
        label: Option<&'static str>,
        span: SourceSpan,
        correlated: CorrelatedSourceSpan,
    ) -> Self {
        CompositeSourceSpan {
            label,
            kind: CompositeKind::Correlated { span, correlated },
        }
    }

    pub fn labels(&self) -> Vec<LabeledSpan> {
        let label = self.label.map(|label| label.to_string());
        match self.kind {
            CompositeKind::Span(ref span) => vec![LabeledSpan::new_with_span(label, span.clone())],
            CompositeKind::Correlated {
                ref span,
                ref correlated,
            } => {
                let mut labels = vec![LabeledSpan::new_with_span(label, span.clone())];
                labels.extend(correlated.labels());
                labels
            }
        }
    }
}

#[derive(Clone, Debug)]
enum CompositeKind {
    Span(SourceSpan),
    Correlated {
        span: SourceSpan,
        correlated: CorrelatedSourceSpan,
    },
}

#[derive(Clone, Debug)]
pub enum CorrelatedSourceSpan {
    Contiguous(SourceSpan),
    Split(SourceSpan, SourceSpan),
}

impl CorrelatedSourceSpan {
    pub fn split_some(left: Option<SourceSpan>, right: SourceSpan) -> Self {
        if let Some(left) = left {
            CorrelatedSourceSpan::Split(left, right)
        }
        else {
            CorrelatedSourceSpan::Contiguous(right)
        }
    }

    pub fn labels(&self) -> Vec<LabeledSpan> {
        let label = Some("here".to_string());
        match self {
            CorrelatedSourceSpan::Contiguous(ref span) => {
                vec![LabeledSpan::new_with_span(label, span.clone())]
            }
            CorrelatedSourceSpan::Split(ref left, ref right) => vec![
                LabeledSpan::new_with_span(label.clone(), left.clone()),
                LabeledSpan::new_with_span(label, right.clone()),
            ],
        }
    }
}

impl From<SourceSpan> for CorrelatedSourceSpan {
    fn from(span: SourceSpan) -> Self {
        CorrelatedSourceSpan::Contiguous(span)
    }
}

#[derive(Clone, Debug, Diagnostic, Error)]
#[diagnostic(code(glob::semantic_literal), severity(warning))]
#[error("\"{literal}\" has been interpreted as a literal with no semantics")]
pub struct SemanticLiteralWarning<'t> {
    #[source_code]
    expression: Cow<'t, str>,
    literal: Cow<'t, str>,
    #[label("here")]
    span: SourceSpan,
}

#[derive(Clone, Debug, Diagnostic, Error)]
#[diagnostic(code(glob::terminating_separator), severity(warning))]
#[error("terminating separator may discard matches")]
pub struct TerminatingSeparatorWarning<'t> {
    #[source_code]
    expression: Cow<'t, str>,
    #[label("here")]
    span: SourceSpan,
}

pub fn diagnostics<'t>(tokenized: &'t Tokenized<'t, Annotation>) -> Vec<Box<dyn Diagnostic + 't>> {
    None.into_iter()
        .chain(
            token::components(tokenized.tokens().iter())
                .flat_map(|component| component.literal().map(|literal| (component, literal)))
                .filter(|(_, literal)| literal.is_semantic_literal())
                .map(|(component, literal)| {
                    Box::new(SemanticLiteralWarning {
                        expression: tokenized.expression().clone(),
                        literal: literal.text().clone(),
                        span: component
                            .tokens()
                            .iter()
                            .map(|token| SourceSpan::from(*token.annotation()))
                            .fold1(|left, right| left.union(&right))
                            .expect("no tokens in component"),
                    }) as Box<dyn Diagnostic>
                }),
        )
        .chain(tokenized.tokens().last().into_iter().flat_map(|token| {
            matches!(token.kind(), TokenKind::Separator).then(|| {
                Box::new(TerminatingSeparatorWarning {
                    expression: tokenized.expression().clone(),
                    span: token.annotation().clone().into(),
                }) as Box<dyn Diagnostic>
            })
        }))
        .collect()
}
