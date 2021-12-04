//! Rules and limitations for token sequences.
//!
//! This module provides the `check` function, which examines a token sequence
//! and emits an error if the sequence violates rules. Rules are invariants that
//! are difficult or impossible to enforce when parsing text and primarily
//! detect and reject token sequences that produce anomalous, meaningless, or
//! unexpected globs (regular expressions) when compiled.
//!
//! Most rules concern alternatives, which have complex interactions with
//! neighboring tokens.

use itertools::Itertools as _;
#[cfg(feature = "diagnostics-error")]
use miette::{Diagnostic, LabeledSpan, SourceCode};
use std::borrow::Cow;
#[cfg(feature = "diagnostics-error")]
use std::fmt::Display;
use thiserror::Error;

#[cfg(feature = "diagnostics-error")]
use crate::diagnostics::error::{CompositeSourceSpan, CorrelatedSourceSpan, SourceSpanExt as _};
use crate::token::{Annotation, Token, TokenKind};
use crate::{IteratorExt as _, SliceExt as _, Terminals};

#[derive(Debug, Error)]
#[error("malformed glob expression: {kind}")]
pub struct RuleError<'t> {
    expression: Cow<'t, str>,
    kind: ErrorKind,
    #[cfg(feature = "diagnostics-error")]
    span: CompositeSourceSpan,
}

impl<'t> RuleError<'t> {
    fn new(
        expression: &'t str,
        kind: ErrorKind,
        #[cfg(feature = "diagnostics-error")] span: CompositeSourceSpan,
    ) -> Self {
        RuleError {
            expression: expression.into(),
            kind,
            #[cfg(feature = "diagnostics-error")]
            span,
        }
    }

    pub fn into_owned(self) -> RuleError<'static> {
        let RuleError {
            expression,
            kind,
            #[cfg(feature = "diagnostics-error")]
            span,
        } = self;
        RuleError {
            expression: expression.into_owned().into(),
            kind,
            #[cfg(feature = "diagnostics-error")]
            span,
        }
    }

    pub fn expression(&self) -> &str {
        self.expression.as_ref()
    }
}

#[cfg(feature = "diagnostics-error")]
impl<'t> Diagnostic for RuleError<'t> {
    fn code<'a>(&'a self) -> Option<Box<dyn 'a + Display>> {
        Some(Box::new(String::from("glob::rule")))
    }

    fn source_code(&self) -> Option<&dyn SourceCode> {
        Some(&self.expression)
    }

    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan>>> {
        Some(Box::new(self.span.labels().into_iter()))
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
enum ErrorKind {
    #[error("rooted sub-glob in group")]
    RootedSubGlob,
    #[error("singular tree wildcard `**` in group")]
    SingularTree,
    #[error("singular zero-or-more wildcard `*` or `$` in group")]
    SingularZeroOrMore,
    #[error("adjacent component boundaries `/` or `**`")]
    AdjacentBoundary,
    #[error("adjacent zero-or-more wildcards `*` or `$`")]
    AdjacentZeroOrMore,
}

pub fn check<'t, 'i, I>(expression: &'t str, tokens: I) -> Result<(), RuleError<'t>>
where
    I: IntoIterator<Item = &'i Token<'t, Annotation>>,
    I::IntoIter: Clone,
    't: 'i,
{
    let tokens = tokens.into_iter();
    boundary(expression, tokens.clone())?;
    group(expression, tokens)?;
    Ok(())
}

fn boundary<'t, 'i, I>(expression: &'t str, tokens: I) -> Result<(), RuleError<'t>>
where
    I: IntoIterator<Item = &'i Token<'t, Annotation>>,
    't: 'i,
{
    #[cfg_attr(not(feature = "diagnostics-error"), allow(unused))]
    if let Some((left, right)) = tokens
        .into_iter()
        .tuple_windows::<(_, _)>()
        .find(|(left, right)| left.is_component_boundary() && right.is_component_boundary())
        .map(|(left, right)| (left.annotation(), right.annotation()))
    {
        Err(RuleError::new(
            expression,
            ErrorKind::AdjacentBoundary,
            #[cfg(feature = "diagnostics-error")]
            CompositeSourceSpan::span(Some("here"), left.union(right)),
        ))
    }
    else {
        Ok(())
    }
}

fn group<'t, 'i, I>(expression: &'t str, tokens: I) -> Result<(), RuleError<'t>>
where
    I: IntoIterator<Item = &'i Token<'t, Annotation>>,
    't: 'i,
{
    use crate::token::TokenKind::{Separator, Wildcard};
    use crate::token::Wildcard::{Tree, ZeroOrMore};
    use crate::Terminals::{Only, StartEnd};

    struct CorrelatedError {
        kind: ErrorKind,
        #[cfg(feature = "diagnostics-error")]
        span: CorrelatedSourceSpan,
    }

    impl CorrelatedError {
        #[cfg_attr(not(feature = "diagnostics-error"), allow(unused))]
        fn new(
            kind: ErrorKind,
            outer: Option<&Token<Annotation>>,
            inner: &Token<Annotation>,
        ) -> Self {
            CorrelatedError {
                kind,
                #[cfg(feature = "diagnostics-error")]
                span: CorrelatedSourceSpan::split_some(
                    outer.map(Token::annotation).cloned(),
                    inner.annotation().clone(),
                ),
            }
        }
    }

    #[derive(Clone, Copy, Default)]
    struct Outer<'t, 'i> {
        left: Option<&'i Token<'t, Annotation>>,
        right: Option<&'i Token<'t, Annotation>>,
    }

    impl<'t, 'i> Outer<'t, 'i> {
        pub fn push(
            self,
            left: Option<&'i Token<'t, Annotation>>,
            right: Option<&'i Token<'t, Annotation>>,
        ) -> Self {
            Outer {
                left: left.or(self.left),
                right: right.or(self.right),
            }
        }
    }

    fn has_preceding_component_boundary<'t>(token: Option<&'t Token<'t, Annotation>>) -> bool {
        token
            .map(|token| token.has_preceding_token_with(&mut |token| token.is_component_boundary()))
            .unwrap_or(false)
    }

    fn has_terminating_component_boundary<'t>(token: Option<&'t Token<'t, Annotation>>) -> bool {
        token
            .map(|token| {
                token.has_terminating_token_with(&mut |token| token.is_component_boundary())
            })
            .unwrap_or(false)
    }

    fn has_preceding_zom_token<'t>(token: Option<&'t Token<'t, Annotation>>) -> bool {
        token
            .map(|token| {
                token.has_preceding_token_with(&mut |token| {
                    matches!(token.kind(), Wildcard(ZeroOrMore(_)))
                })
            })
            .unwrap_or(false)
    }

    fn has_terminating_zom_token<'t>(token: Option<&'t Token<'t, Annotation>>) -> bool {
        token
            .map(|token| {
                token.has_terminating_token_with(&mut |token| {
                    matches!(token.kind(), Wildcard(ZeroOrMore(_)))
                })
            })
            .unwrap_or(false)
    }

    #[cfg_attr(not(feature = "diagnostics-error"), allow(unused))]
    fn diagnose<'t, 'i>(
        token: &'i Token<'t, Annotation>,
        expression: &'t str,
        label: &'static str,
    ) -> impl 'i + Copy + Fn(CorrelatedError) -> RuleError<'t>
    where
        't: 'i,
    {
        move |CorrelatedError {
                  kind,
                  #[cfg(feature = "diagnostics-error")]
                  span,
              }| {
            RuleError::new(
                expression,
                kind,
                #[cfg(feature = "diagnostics-error")]
                CompositeSourceSpan::correlated(Some(label), token.annotation().clone(), span),
            )
        }
    }

    fn recurse<'t, 'i, I>(
        expression: &'t str,
        tokens: I,
        outer: Outer<'t, 'i>,
    ) -> Result<(), RuleError<'t>>
    where
        I: IntoIterator<Item = &'i Token<'t, Annotation>>,
        't: 'i,
    {
        for (left, token, right) in tokens
            .into_iter()
            .adjacent()
            .map(|adjacency| adjacency.into_tuple())
        {
            match token.kind() {
                TokenKind::Alternative(ref alternative) => {
                    let outer = outer.push(left, right);
                    let diagnose = diagnose(token, expression, "in this alternative");
                    for tokens in alternative.branches() {
                        if let Some(terminals) = tokens.terminals() {
                            check_group(terminals, outer).map_err(diagnose)?;
                            check_group_alternative(terminals, outer).map_err(diagnose)?;
                        }
                        recurse(expression, tokens.iter(), outer)?;
                    }
                }
                TokenKind::Repetition(ref repetition) => {
                    let outer = outer.push(left, right);
                    let diagnose = diagnose(token, expression, "in this repetition");
                    let tokens = repetition.tokens();
                    if let Some(terminals) = tokens.terminals() {
                        check_group(terminals, outer).map_err(diagnose)?;
                        check_group_repetition(terminals, outer, repetition.bounds())
                            .map_err(diagnose)?;
                    }
                    recurse(expression, tokens.iter(), outer)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn check_group<'t>(
        terminals: Terminals<&Token<Annotation>>,
        outer: Outer<'t, 't>,
    ) -> Result<(), CorrelatedError> {
        let Outer { left, right } = outer;
        match terminals.map(|token| (token, token.kind())) {
            // The group is preceded by component boundaries; disallow leading
            // separators.
            //
            // For example, `foo/{bar,/}`.
            Only((inner, Separator)) | StartEnd((inner, Separator), _)
                if has_terminating_component_boundary(left) =>
            {
                Err(CorrelatedError::new(
                    ErrorKind::AdjacentBoundary,
                    left,
                    inner,
                ))
            }
            // The group is followed by component boundaries; disallow trailing
            // separators.
            //
            // For example, `{foo,/}/bar`.
            Only((inner, Separator)) | StartEnd(_, (inner, Separator))
                if has_preceding_component_boundary(right) =>
            {
                Err(CorrelatedError::new(
                    ErrorKind::AdjacentBoundary,
                    right,
                    inner,
                ))
            }
            // Disallow singular tree tokens.
            //
            // For example, `{foo,bar,**}`.
            Only((inner, Wildcard(Tree { .. }))) => {
                Err(CorrelatedError::new(ErrorKind::SingularTree, None, inner))
            }
            // The group is preceded by component boundaries; disallow leading
            // tree tokens.
            //
            // For example, `foo/{bar,**/baz}`.
            StartEnd((inner, Wildcard(Tree { .. })), _)
                if has_terminating_component_boundary(left) =>
            {
                Err(CorrelatedError::new(
                    ErrorKind::AdjacentBoundary,
                    left,
                    inner,
                ))
            }
            // The group is followed by component boundaries; disallow trailing
            // tree tokens.
            //
            // For example, `{foo,bar/**}/baz`.
            StartEnd(_, (inner, Wildcard(Tree { .. })))
                if has_preceding_component_boundary(right) =>
            {
                Err(CorrelatedError::new(
                    ErrorKind::AdjacentBoundary,
                    right,
                    inner,
                ))
            }
            // The group is prefixed by a zero-or-more token; disallow leading
            // zero-or-more tokens.
            //
            // For example, `foo*{bar,*,baz}`.
            Only((inner, Wildcard(ZeroOrMore(_))))
            | StartEnd((inner, Wildcard(ZeroOrMore(_))), _)
                if has_terminating_zom_token(left) =>
            {
                Err(CorrelatedError::new(
                    ErrorKind::AdjacentZeroOrMore,
                    left,
                    inner,
                ))
            }
            // The group is followed by a zero-or-more token; disallow trailing
            // zero-or-more tokens.
            //
            // For example, `{foo,*,bar}*baz`.
            Only((inner, Wildcard(ZeroOrMore(_))))
            | StartEnd(_, (inner, Wildcard(ZeroOrMore(_))))
                if has_preceding_zom_token(right) =>
            {
                Err(CorrelatedError::new(
                    ErrorKind::AdjacentZeroOrMore,
                    right,
                    inner,
                ))
            }
            _ => Ok(()),
        }
    }

    fn check_group_alternative<'t>(
        terminals: Terminals<&Token<Annotation>>,
        outer: Outer<'t, 't>,
    ) -> Result<(), CorrelatedError> {
        let Outer { left, .. } = outer;
        match terminals.map(|token| (token, token.kind())) {
            // The alternative is preceded by a termination; disallow rooted
            // sub-globs.
            //
            // For example, `{foo,/}` or `{foo,/bar}`.
            Only((inner, Separator)) | StartEnd((inner, Separator), _) if left.is_none() => {
                Err(CorrelatedError::new(ErrorKind::RootedSubGlob, left, inner))
            }
            // The alternative is preceded by a termination; disallow rooted
            // sub-globs.
            //
            // For example, `{/**/foo,bar}`.
            Only((inner, Wildcard(Tree { is_rooted: true })))
            | StartEnd((inner, Wildcard(Tree { is_rooted: true })), _)
                if left.is_none() =>
            {
                Err(CorrelatedError::new(ErrorKind::RootedSubGlob, left, inner))
            }
            _ => Ok(()),
        }
    }

    fn check_group_repetition<'t>(
        terminals: Terminals<&Token<Annotation>>,
        outer: Outer<'t, 't>,
        bounds: (usize, Option<usize>),
    ) -> Result<(), CorrelatedError> {
        let Outer { left, .. } = outer;
        let (lower, _) = bounds;
        match terminals.map(|token| (token, token.kind())) {
            // The repetition is preceded by a termination; disallow rooted
            // sub-globs with a zero lower bound.
            //
            // For example, `</foo:0,>`.
            Only((inner, Separator)) | StartEnd((inner, Separator), _)
                if left.is_none() && lower == 0 =>
            {
                Err(CorrelatedError::new(ErrorKind::RootedSubGlob, left, inner))
            }
            // The repetition is preceded by a termination; disallow rooted
            // sub-globs with a zero lower bound.
            //
            // For example, `</**/foo>`.
            Only((inner, Wildcard(Tree { is_rooted: true })))
            | StartEnd((inner, Wildcard(Tree { is_rooted: true })), _)
                if left.is_none() && lower == 0 =>
            {
                Err(CorrelatedError::new(ErrorKind::RootedSubGlob, left, inner))
            }
            // The repetition begins and ends with a separator.
            //
            // For example, `</foo/bar/:1,>`.
            StartEnd((left, _), (right, _))
                if left.is_component_boundary() && right.is_component_boundary() =>
            {
                Err(CorrelatedError::new(
                    ErrorKind::AdjacentBoundary,
                    Some(left),
                    right,
                ))
            }
            // The repetition is a singular separator.
            //
            // For example, `</:1,>`.
            Only((token, Separator)) => Err(CorrelatedError::new(
                ErrorKind::AdjacentBoundary,
                None,
                token,
            )),
            // The repetition is a singular zero-or-more wildcard.
            //
            // For example, `<*:1,>`.
            Only((token, Wildcard(ZeroOrMore(_)))) => Err(CorrelatedError::new(
                ErrorKind::SingularZeroOrMore,
                None,
                token,
            )),
            _ => Ok(()),
        }
    }

    recurse(expression, tokens, Default::default())
}
