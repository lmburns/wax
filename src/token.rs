mod supreme {
    //! Re-exported items from `nom_supreme`.
    //!
    //! This module collapses `nom_supreme` and exposes its items using similar
    //! names to `nom`. Parsers are used via this module such that they resemble
    //! the use of `nom` parsers. See the `parse` function.

    #[allow(unused_imports)]
    pub(crate) use nom_supreme::{
        error::{BaseErrorKind, ErrorTree, StackContext},
        multi::{collect_separated_terminated as many1, parse_separated_terminated as fold1},
        parser_ext::ParserExt,
        tag::complete::tag,
    };
}

use itertools::Itertools as _;
#[cfg(feature = "diagnostics-report")]
use miette::{self, Diagnostic, LabeledSpan, SourceCode};
use pori::{Located, Location, Stateful};
use smallvec::{smallvec, SmallVec};
use std::{
    borrow::Cow,
    cmp,
    fmt::{self, Display, Formatter},
    mem,
    ops::{Bound, Deref, RangeBounds},
    path::{PathBuf, MAIN_SEPARATOR},
    str::FromStr,
};
use supreme::{BaseErrorKind, StackContext};
use thiserror::Error;

#[cfg(any(feature = "diagnostics-inspect", feature = "diagnostics-report"))]
use crate::diagnostics::Span;
use crate::{SliceExt as _, StrExt as _, Terminals, PATHS_ARE_CASE_INSENSITIVE};

#[cfg(any(feature = "diagnostics-inspect", feature = "diagnostics-report"))]
pub(crate) type Annotation = Span;
#[cfg(all(
    not(feature = "diagnostics-inspect"),
    not(feature = "diagnostics-report"),
))]
pub(crate) type Annotation = ();

type Expression<'i> = Located<'i, str>;
type Input<'i> = Stateful<Expression<'i>, ParserState>;
type ErrorTree<'i> = supreme::ErrorTree<Input<'i>>;
type ErrorMode<'t> = nom::Err<ErrorTree<'t>>;

#[derive(Clone, Debug)]
struct ErrorLocation {
    location: usize,
    context:  String,
}

impl<'e, 'i> From<ErrorEntry<'e, Input<'i>>> for ErrorLocation {
    fn from(entry: ErrorEntry<'e, Input<'i>>) -> Self {
        Self {
            location: entry.input.location(),
            context:  entry.context.to_string(),
        }
    }
}

#[cfg(feature = "diagnostics-report")]
impl From<ErrorLocation> for LabeledSpan {
    fn from(location: ErrorLocation) -> Self {
        let ErrorLocation { location, context } = location;
        LabeledSpan::new(Some(context), location, 1)
    }
}

impl Display for ErrorLocation {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "at offset {}: {}", self.location, self.context)
    }
}

#[derive(Clone, Debug)]
struct ErrorEntry<'e, I> {
    depth:   usize,
    input:   &'e I,
    context: ErrorContext<'e>,
}

#[derive(Clone, Debug)]
enum ErrorContext<'e> {
    Kind(&'e BaseErrorKind),
    Stack(&'e StackContext),
}

impl Display for ErrorContext<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ErrorContext::Kind(kind) => match kind {
                BaseErrorKind::Expected(_) | BaseErrorKind::Kind(_) => write!(f, "{}", kind),
                // Omit any "external error" prefix as seen in the `Display`
                // implementation of `BaseErrorKind`.
                BaseErrorKind::External(error) => write!(f, "{}", error),
            },
            ErrorContext::Stack(stack) => write!(f, "{}", stack),
        }
    }
}

trait ErrorTreeExt<I> {
    fn for_each<F>(&self, f: F)
    where
        F: FnMut(ErrorEntry<I>);

    fn bounding_error_locations(&self) -> (Vec<ErrorLocation>, Vec<ErrorLocation>);
}

impl<'i> ErrorTreeExt<Input<'i>> for ErrorTree<'i> {
    fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(ErrorEntry<Input<'i>>),
    {
        fn recurse<'i, F>(tree: &'_ ErrorTree<'i>, depth: usize, f: &mut F)
        where
            F: FnMut(ErrorEntry<Input<'i>>),
        {
            match tree {
                ErrorTree::Base { ref location, ref kind } =>
                    f(ErrorEntry { depth, input: location, context: ErrorContext::Kind(kind) }),
                ErrorTree::Stack { ref base, ref contexts } => {
                    for (location, context) in contexts {
                        f(ErrorEntry {
                            depth:   depth + 1,
                            input:   location,
                            context: ErrorContext::Stack(context),
                        });
                    }
                    recurse(base, depth + 1, f);
                },
                ErrorTree::Alt(ref trees) =>
                    for tree in trees {
                        recurse(tree, depth + 1, f);
                    },
            }
        }
        recurse(self, 0, &mut f);
    }

    fn bounding_error_locations(&self) -> (Vec<ErrorLocation>, Vec<ErrorLocation>) {
        let mut min: Option<(usize, Vec<ErrorLocation>)> = None;
        let mut max: Option<(usize, Vec<ErrorLocation>)> = None;
        self.for_each(|entry| {
            let write_if =
                |locations: &mut Option<(_, _)>, push: fn(&_, &_) -> _, reset: fn(&_, &_) -> _| {
                    let locations = locations.get_or_insert_with(|| (entry.depth, vec![]));
                    if reset(&entry.depth, &locations.0) {
                        *locations = (entry.depth, vec![]);
                    }
                    if push(&entry.depth, &locations.0) {
                        locations.1.push(entry.clone().into());
                    }
                };
            write_if(&mut min, usize::eq, |&depth, &min| depth < min);
            write_if(&mut max, usize::eq, |&depth, &max| depth > max);
        });
        (
            min.map(|(_, locations)| locations).unwrap_or_default(),
            max.map(|(_, locations)| locations).unwrap_or_default(),
        )
    }
}

/// Describes errors that occur when parsing a glob expression.
///
/// Common examples of glob expressions that cannot be parsed are alternative
/// and repetition patterns with missing delimiters and ambiguous patterns, such
/// as `src/***/*.rs` or `{.local,.config/**/*.toml`.
///
/// When the `diagnostics-report` feature is enabled, this error implements the
/// [`Diagnostic`] trait and provides more detailed information about the parse
/// failure.
///
/// [`Diagnostic`]: miette::Diagnostic
#[derive(Clone, Debug, Error)]
#[error("failed to parse glob expression")]
pub struct ParseError<'t> {
    expression: Cow<'t, str>,
    start:      ErrorLocation,
    ends:       Vec<ErrorLocation>,
}

impl<'t> ParseError<'t> {
    fn new(expression: &'t str, error: ErrorMode<'t>) -> Self {
        match error {
            ErrorMode::Incomplete(_) => {
                panic!("unexpected parse error: incomplete input")
            },
            ErrorMode::Error(error) | ErrorMode::Failure(error) => {
                let (starts, ends) = error.bounding_error_locations();
                ParseError {
                    expression: expression.into(),
                    start: starts
                        .into_iter()
                        .next()
                        .expect("expected lower bound error location"),
                    ends,
                }
            },
        }
    }

    /// Clones any borrowed data into an owning instance.
    #[inline]
    #[must_use]
    pub fn into_owned(self) -> ParseError<'static> {
        let ParseError { expression, start, ends } = self;
        ParseError { expression: expression.into_owned().into(), start, ends }
    }

    /// Gets the glob expression that failed to parse.
    #[inline]
    #[must_use]
    pub fn expression(&self) -> &str {
        self.expression.as_ref()
    }
}

#[cfg(feature = "diagnostics-report")]
#[cfg_attr(docsrs, doc(cfg(feature = "diagnostics-report")))]
impl Diagnostic for ParseError<'_> {
    fn code<'a>(&'a self) -> Option<Box<dyn Display + 'a>> {
        Some(Box::new("wax::glob::parse"))
    }

    fn source_code(&self) -> Option<&dyn SourceCode> {
        Some(&self.expression)
    }

    // Surfacing useful parsing errors is difficult. This code replaces any
    // lower bound errors with a simple label noting the beginning of the
    // parsing error. Details are discarded, because these are typically
    // top-level alternative errors that do not provide any useful insight.
    // Upper bound errors are labeled as-is, though they only sometimes provide
    // useful context.
    fn labels(&self) -> Option<Box<dyn Iterator<Item = LabeledSpan> + '_>> {
        Some(Box::new(
            Some(LabeledSpan::new(
                Some(String::from("starting here")),
                self.start.location,
                1,
            ))
            .into_iter()
            .chain(self.ends.iter().cloned().map(|end| {
                LabeledSpan::new(
                    Some(end.context),
                    self.start.location,
                    end.location.saturating_sub(self.start.location) + 1,
                )
            })),
        ))
    }
}

pub trait IntoTokens<'t>: Sized {
    type Annotation;

    fn into_tokens(self) -> Vec<Token<'t, Self::Annotation>>;
}

#[derive(Clone, Copy, Debug, Default)]
struct ParserState {
    flags:         FlagState,
    subexpression: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FlagState {
    is_case_insensitive: bool,
}

impl Default for FlagState {
    fn default() -> Self {
        Self { is_case_insensitive: PATHS_ARE_CASE_INSENSITIVE }
    }
}

#[derive(Clone, Copy, Debug)]
enum FlagToggle {
    CaseInsensitive(bool),
}

#[derive(Clone, Debug)]
pub(crate) struct Tokenized<'t, A = Annotation> {
    expression: Cow<'t, str>,
    tokens:     Vec<Token<'t, A>>,
}

impl<'t, A> Tokenized<'t, A> {
    pub(crate) fn into_owned(self) -> Tokenized<'static, A> {
        let Tokenized { expression, tokens } = self;
        Tokenized {
            expression: expression.into_owned().into(),
            tokens:     tokens.into_iter().map(Token::into_owned).collect(),
        }
    }

    pub(crate) fn partition(mut self) -> (PathBuf, Self) {
        // Get the invariant prefix for the token sequence.
        let prefix = invariant_prefix_path(self.tokens.iter()).unwrap_or_else(PathBuf::new);

        // Drain invariant tokens from the beginning of the token sequence and
        // unroot any tokens at the beginning of the sequence (tree wildcards).
        self.tokens
            .drain(0..invariant_prefix_upper_bound(&self.tokens));
        self.tokens.first_mut().map(Token::unroot);

        (prefix, self)
    }

    pub(crate) const fn expression(&self) -> &Cow<'t, str> {
        &self.expression
    }

    pub(crate) fn tokens(&self) -> &[Token<'t, A>] {
        &self.tokens
    }
}

impl<'t, A> IntoTokens<'t> for Tokenized<'t, A> {
    type Annotation = A;

    fn into_tokens(self) -> Vec<Token<'t, Self::Annotation>> {
        let Tokenized { tokens, .. } = self;
        tokens
    }
}

#[derive(Clone, Debug)]
pub struct Token<'t, A = Annotation> {
    kind:       TokenKind<'t, A>,
    annotation: A,
}

impl<'t, A> Token<'t, A> {
    const fn new(kind: TokenKind<'t, A>, annotation: A) -> Self {
        Token { kind, annotation }
    }

    pub fn into_owned(self) -> Token<'static, A> {
        let Token { kind, annotation } = self;
        Token { kind: kind.into_owned(), annotation }
    }

    pub fn unannotate(self) -> Token<'t, ()> {
        let Token { kind, .. } = self;
        Token { kind: kind.unannotate(), annotation: () }
    }

    pub fn unroot(&mut self) -> bool {
        self.kind.unroot()
    }

    pub fn kind(&self) -> &TokenKind<'t, A> {
        self.as_ref()
    }

    pub fn annotation(&self) -> &A {
        self.as_ref()
    }

    pub fn is_rooted(&self) -> bool {
        self.has_preceding_token_with(&mut |token| {
            matches!(
                token.kind(),
                TokenKind::Separator | TokenKind::Wildcard(Wildcard::Tree { is_rooted: true })
            )
        })
    }

    pub fn has_component_boundary(&self) -> bool {
        self.has_token_with(&mut |token| token.is_component_boundary())
    }

    pub fn has_token_with(&self, f: &mut impl FnMut(&Token<'t, A>) -> bool) -> bool {
        match self.kind() {
            TokenKind::Alternative(ref alternative) => alternative.has_token_with(f),
            TokenKind::Repetition(ref repetition) => repetition.has_token_with(f),
            TokenKind::Class(_)
            | TokenKind::Literal(_)
            | TokenKind::Separator
            | TokenKind::Wildcard(_) => f(self),
        }
    }

    pub fn has_preceding_token_with(&self, f: &mut impl FnMut(&Token<'t, A>) -> bool) -> bool {
        match self.kind() {
            TokenKind::Alternative(ref alternative) => alternative.has_preceding_token_with(f),
            TokenKind::Repetition(ref repetition) => repetition.has_preceding_token_with(f),
            TokenKind::Class(_)
            | TokenKind::Literal(_)
            | TokenKind::Separator
            | TokenKind::Wildcard(_) => f(self),
        }
    }

    pub fn has_terminating_token_with(&self, f: &mut impl FnMut(&Token<'t, A>) -> bool) -> bool {
        match self.kind() {
            TokenKind::Alternative(ref alternative) => alternative.has_terminating_token_with(f),
            TokenKind::Repetition(ref repetition) => repetition.has_terminating_token_with(f),
            TokenKind::Class(_)
            | TokenKind::Literal(_)
            | TokenKind::Separator
            | TokenKind::Wildcard(_) => f(self),
        }
    }
}

impl<'t, A> AsRef<TokenKind<'t, A>> for Token<'t, A> {
    fn as_ref(&self) -> &TokenKind<'t, A> {
        &self.kind
    }
}

impl<A> AsRef<A> for Token<'_, A> {
    fn as_ref(&self) -> &A {
        &self.annotation
    }
}

impl<'t, A> Deref for Token<'t, A> {
    type Target = TokenKind<'t, A>;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'t> From<TokenKind<'t, ()>> for Token<'t, ()> {
    fn from(kind: TokenKind<'t, ()>) -> Self {
        Token { kind, annotation: () }
    }
}

#[derive(Clone, Debug)]
pub enum TokenKind<'t, A = ()> {
    Alternative(Alternative<'t, A>),
    Class(Class),
    Literal(Literal<'t>),
    Repetition(Repetition<'t, A>),
    Separator,
    Wildcard(Wildcard),
    // MatchOneNotAnother,
}

impl<'t, A> TokenKind<'t, A> {
    pub fn into_owned(self) -> TokenKind<'static, A> {
        match self {
            TokenKind::Alternative(alternative) => alternative.into_owned().into(),
            TokenKind::Class(class) => TokenKind::Class(class),
            TokenKind::Literal(Literal { text, is_case_insensitive }) =>
                TokenKind::Literal(Literal { text: text.into_owned().into(), is_case_insensitive }),
            TokenKind::Repetition(repetition) => repetition.into_owned().into(),
            TokenKind::Separator => TokenKind::Separator,
            TokenKind::Wildcard(wildcard) => TokenKind::Wildcard(wildcard),
            // TokenKind::MatchOneNotAnother => TokenKind::MatchOneNotAnother,
        }
    }

    pub fn unannotate(self) -> TokenKind<'t, ()> {
        match self {
            TokenKind::Alternative(alternative) => TokenKind::Alternative(alternative.unannotate()),
            TokenKind::Class(class) => TokenKind::Class(class),
            TokenKind::Literal(literal) => TokenKind::Literal(literal),
            TokenKind::Repetition(repetition) => TokenKind::Repetition(repetition.unannotate()),
            TokenKind::Separator => TokenKind::Separator,
            TokenKind::Wildcard(wildcard) => TokenKind::Wildcard(wildcard),
            // TokenKind::MatchOneNotAnother => TokenKind::MatchOneNotAnother,
        }
    }

    pub fn unroot(&mut self) -> bool {
        if let TokenKind::Wildcard(Wildcard::Tree { ref mut is_rooted }) = self {
            mem::replace(is_rooted, false)
        } else {
            false
        }
    }

    pub fn to_invariant_string(&self) -> Option<Cow<str>> {
        match self {
            TokenKind::Alternative(ref alternative) => alternative.to_invariant_string(),
            TokenKind::Class(ref class) => class.to_invariant_string(),
            TokenKind::Literal(ref literal) => literal.to_invariant_string(),
            TokenKind::Repetition(ref repetition) => repetition.to_invariant_string(),
            TokenKind::Separator => Some(MAIN_SEPARATOR.to_string().into()),
            TokenKind::Wildcard(_) => None,
            // TokenKind::MatchOneNotAnother => None,
        }
    }

    pub const fn has_sub_tokens(&self) -> bool {
        // It is not necessary to detect empty branches or sub-expressions.
        matches!(self, TokenKind::Alternative(_) | TokenKind::Repetition(_))
    }

    pub const fn is_component_boundary(&self) -> bool {
        matches!(
            self,
            TokenKind::Separator | TokenKind::Wildcard(Wildcard::Tree { .. })
        )
    }

    #[cfg_attr(not(feature = "diagnostics-inspect"), allow(unused))]
    pub const fn is_capturing(&self) -> bool {
        use TokenKind::{Alternative, Class, Repetition, Wildcard};

        matches!(
            self,
            Alternative(_) | Class(_) | Repetition(_) | Wildcard(_),
        )
    }
}

impl<'t, A> From<Alternative<'t, A>> for TokenKind<'t, A> {
    fn from(alternative: Alternative<'t, A>) -> Self {
        TokenKind::Alternative(alternative)
    }
}

impl<A> From<Class> for TokenKind<'_, A> {
    fn from(class: Class) -> Self {
        TokenKind::Class(class)
    }
}

impl<'t, A> From<Repetition<'t, A>> for TokenKind<'t, A> {
    fn from(repetition: Repetition<'t, A>) -> Self {
        TokenKind::Repetition(repetition)
    }
}

impl<A> From<Wildcard> for TokenKind<'static, A> {
    fn from(wildcard: Wildcard) -> Self {
        TokenKind::Wildcard(wildcard)
    }
}

#[derive(Clone, Debug)]
pub struct Alternative<'t, A = ()>(Vec<Vec<Token<'t, A>>>);

impl<'t, A> Alternative<'t, A> {
    pub fn into_owned(self) -> Alternative<'static, A> {
        Alternative(
            self.0
                .into_iter()
                .map(|tokens| tokens.into_iter().map(Token::into_owned).collect())
                .collect(),
        )
    }

    pub fn unannotate(self) -> Alternative<'t, ()> {
        let Alternative(branches) = self;
        Alternative(
            branches
                .into_iter()
                .map(|branch| branch.into_iter().map(super::Token::unannotate).collect())
                .collect(),
        )
    }

    pub const fn branches(&self) -> &Vec<Vec<Token<'t, A>>> {
        &self.0
    }

    pub fn to_invariant_string(&self) -> Option<Cow<str>> {
        match self.0.terminals() {
            Some(Terminals::Only(tokens)) => fold_invariant_strings(tokens.iter()),
            None => Some("".into()),
            _ => None,
        }
    }

    pub fn has_token_with(&self, f: &mut impl FnMut(&Token<'t, A>) -> bool) -> bool {
        self.0
            .iter()
            .any(|tokens| tokens.iter().any(|token| token.has_token_with(f)))
    }

    pub fn has_preceding_token_with(&self, f: &mut impl FnMut(&Token<'t, A>) -> bool) -> bool {
        self.0.iter().any(|tokens| {
            tokens
                .first()
                .map_or(false, |token| token.has_preceding_token_with(f))
        })
    }

    pub fn has_terminating_token_with(&self, f: &mut impl FnMut(&Token<'t, A>) -> bool) -> bool {
        self.0.iter().any(|tokens| {
            tokens
                .last()
                .map_or(false, |token| token.has_terminating_token_with(f))
        })
    }
}

impl<'t, A> From<Vec<Vec<Token<'t, A>>>> for Alternative<'t, A> {
    fn from(alternatives: Vec<Vec<Token<'t, A>>>) -> Self {
        Alternative(alternatives)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Archetype {
    Character(char),
    Range(char, char),
}

impl Archetype {
    // Using a `&self` receiver interacts better with `Cow` and is more
    // consistent with this method on other token types.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_invariant_string(&self) -> Option<Cow<str>> {
        match self {
            Archetype::Character(x) => (!PATHS_ARE_CASE_INSENSITIVE).then(|| x.to_string().into()),
            Archetype::Range(a, b) =>
                ((a == b) && !PATHS_ARE_CASE_INSENSITIVE).then(|| a.to_string().into()),
        }
    }
}

impl From<char> for Archetype {
    fn from(literal: char) -> Self {
        Self::Character(literal)
    }
}

impl From<(char, char)> for Archetype {
    fn from(range: (char, char)) -> Self {
        Self::Range(range.0, range.1)
    }
}

#[derive(Clone, Debug)]
pub struct Class {
    is_negated: bool,
    archetypes: Vec<Archetype>,
}

impl Class {
    pub fn archetypes(&self) -> &[Archetype] {
        &self.archetypes
    }

    pub const fn is_negated(&self) -> bool {
        self.is_negated
    }

    pub fn to_invariant_string(&self) -> Option<Cow<str>> {
        (!self.is_negated)
            .then(|| match self.archetypes.terminals() {
                Some(Terminals::Only(archetype)) => archetype.to_invariant_string(),
                None => Some("".into()),
                _ => None,
            })
            .flatten()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Evaluation {
    Eager,
    Lazy,
}

#[derive(Clone, Debug)]
pub struct Literal<'t> {
    text:                Cow<'t, str>,
    is_case_insensitive: bool,
}

impl Literal<'_> {
    pub fn text(&self) -> &str {
        self.text.as_ref()
    }

    pub fn to_invariant_string(&self) -> Option<Cow<str>> {
        (!self.has_variant_casing()).then(|| self.text.clone())
    }

    pub const fn is_case_insensitive(&self) -> bool {
        self.is_case_insensitive
    }

    pub fn has_variant_casing(&self) -> bool {
        // If path case sensitivity agrees with the literal case sensitivity,
        // then the literal is not variant. Otherwise, the literal is variant if
        // it contains characters with casing.
        (PATHS_ARE_CASE_INSENSITIVE != self.is_case_insensitive) && self.text.has_casing()
    }
}

#[derive(Clone, Debug)]
pub struct Repetition<'t, A = ()> {
    tokens: Vec<Token<'t, A>>,
    lower:  usize,
    step:   Option<usize>,
}

impl<'t, A> Repetition<'t, A> {
    fn new(tokens: Vec<Token<'t, A>>, bounds: impl RangeBounds<usize>) -> Option<Self> {
        let lower = match bounds.start_bound() {
            Bound::Included(lower) => *lower,
            Bound::Excluded(lower) => *lower + 1,
            Bound::Unbounded => 0,
        };
        let upper = match bounds.end_bound() {
            Bound::Included(upper) => Some(*upper),
            Bound::Excluded(upper) => Some(cmp::max(*upper, 1) - 1),
            Bound::Unbounded => None,
        };
        match upper {
            Some(upper) =>
                (upper >= lower).then(|| Repetition { tokens, lower, step: Some(upper - lower) }),
            None => Some(Repetition { tokens, lower, step: None }),
        }
    }

    pub fn into_owned(self) -> Repetition<'static, A> {
        let Repetition { tokens, lower, step } = self;
        Repetition {
            tokens: tokens.into_iter().map(Token::into_owned).collect(),
            lower,
            step,
        }
    }

    pub fn unannotate(self) -> Repetition<'t, ()> {
        let Repetition { tokens, lower, step } = self;
        Repetition {
            tokens: tokens.into_iter().map(super::Token::unannotate).collect(),
            lower,
            step,
        }
    }

    pub const fn tokens(&self) -> &Vec<Token<'t, A>> {
        &self.tokens
    }

    pub fn bounds(&self) -> (usize, Option<usize>) {
        (self.lower, self.step.map(|step| self.lower + step))
    }

    pub fn to_invariant_string(&self) -> Option<Cow<str>> {
        matches!(self.step, Some(0))
            .then(|| {
                fold_invariant_strings(self.tokens.iter())
                    .map(|string| string.repeat(self.lower).into())
            })
            .flatten()
    }

    pub fn has_token_with(&self, f: &mut impl FnMut(&Token<'t, A>) -> bool) -> bool {
        self.tokens.iter().any(|token| token.has_token_with(f))
    }

    pub fn has_preceding_token_with(&self, f: &mut impl FnMut(&Token<'t, A>) -> bool) -> bool {
        self.tokens
            .first()
            .map_or(false, |token| token.has_preceding_token_with(f))
    }

    pub fn has_terminating_token_with(&self, f: &mut impl FnMut(&Token<'t, A>) -> bool) -> bool {
        self.tokens
            .last()
            .map_or(false, |token| token.has_terminating_token_with(f))
    }
}

#[derive(Clone, Debug)]
pub enum Wildcard {
    One,
    ZeroOrMore(Evaluation),
    Tree { is_rooted: bool },
}

#[derive(Clone, Debug)]
pub(crate) struct LiteralSequence<'i, 't>(SmallVec<[&'i Literal<'t>; 4]>);

impl<'i, 't> LiteralSequence<'i, 't> {
    pub(crate) fn literals(&self) -> &[&'i Literal<'t>] {
        self.0.as_ref()
    }

    pub(crate) fn text(&self) -> Cow<'t, str> {
        if self.literals().len() == 1 {
            self.literals()
                .first()
                .expect("failed to get first item")
                .text
                .clone()
        } else {
            self.literals()
                .iter()
                .map(|literal| &literal.text)
                .join("")
                .into()
        }
    }

    #[cfg(any(unix, windows))]
    pub(crate) fn is_semantic_literal(&self) -> bool {
        matches!(self.text().as_ref(), "." | "..")
    }

    #[cfg(not(any(unix, windows)))]
    pub(crate) fn is_semantic_literal(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub(crate) struct Component<'i, 't, A = ()>(SmallVec<[&'i Token<'t, A>; 4]>);

impl<'i, 't, A> Component<'i, 't, A> {
    pub(crate) fn tokens(&self) -> &[&'i Token<'t, A>] {
        self.0.as_ref()
    }

    pub(crate) fn literal(&self) -> Option<LiteralSequence<'i, 't>> {
        // This predicate is more easily expressed with `all`, but `any` is used
        // here, because it returns `false` for empty iterators and in that case
        // this function should return `None`.
        (!self
            .tokens()
            .iter()
            .any(|token| !matches!(token.kind(), TokenKind::Literal(_))))
        .then(|| {
            LiteralSequence(
                self.tokens()
                    .iter()
                    .map(|token| {
                        if let TokenKind::Literal(ref literal) = token.kind() {
                            literal
                        } else {
                            unreachable!()
                        }
                    })
                    .collect(),
            )
        })
    }

    pub(crate) fn to_invariant_string(&self) -> Option<Cow<str>> {
        fold_invariant_strings(self.0.iter().copied())
    }
}

impl<A> Clone for Component<'_, '_, A> {
    fn clone(&self) -> Self {
        Component(self.0.clone())
    }
}

pub(crate) fn any<'t, A, I>(tokens: I) -> Token<'t, ()>
where
    I: IntoIterator,
    I::Item: IntoIterator<Item = Token<'t, A>>,
{
    Token {
        kind:       Alternative(
            tokens
                .into_iter()
                .map(|tokens| tokens.into_iter().map(super::Token::unannotate).collect())
                .collect(),
        )
        .into(),
        annotation: (),
    }
}

pub(crate) fn components<'i, 't, A, I>(tokens: I) -> impl Iterator<Item = Component<'i, 't, A>>
where
    't: 'i,
    A: 't,
    I: IntoIterator<Item = &'i Token<'t, A>>,
    I::IntoIter: Clone,
{
    tokens.into_iter().batching(|tokens| {
        let mut first = tokens.next();
        while matches!(first.map(Token::kind), Some(TokenKind::Separator)) {
            first = tokens.next();
        }
        first.map(|first| match first.kind() {
            TokenKind::Wildcard(Wildcard::Tree { .. }) => Component(smallvec![first]),
            _ => Component(
                Some(first)
                    .into_iter()
                    .chain(tokens.take_while_ref(|token| !token.is_component_boundary()))
                    .collect(),
            ),
        })
    })
}

// TODO: This implementation allocates many `Vec`s.
pub(crate) fn literals<'i, 't, A, I>(
    tokens: I,
) -> impl Iterator<Item = (Component<'i, 't, A>, LiteralSequence<'i, 't>)>
where
    't: 'i,
    A: 't,
    I: IntoIterator<Item = &'i Token<'t, A>>,
    I::IntoIter: Clone,
{
    components(tokens).flat_map(|component| {
        if let Some(literal) = component.literal() {
            vec![(component, literal)]
        } else {
            component
                .tokens()
                .iter()
                .filter_map(|token| match token.kind() {
                    TokenKind::Alternative(ref alternative) => Some(
                        alternative
                            .branches()
                            .iter()
                            .flat_map(literals)
                            .collect::<Vec<_>>(),
                    ),
                    TokenKind::Repetition(ref repetition) =>
                        Some(literals(repetition.tokens()).collect::<Vec<_>>()),
                    _ => None,
                })
                .flatten()
                .collect::<Vec<_>>()
        }
    })
}

// TODO: Is there some way to unify this with `invariant_prefix_upper_bound`?
pub(crate) fn invariant_prefix_path<'t, A, I>(tokens: I) -> Option<PathBuf>
where
    A: 't,
    I: IntoIterator<Item = &'t Token<'t, A>>,
    I::IntoIter: Clone,
{
    let mut tokens = tokens.into_iter().peekable();
    let mut prefix = String::new();
    if tokens
        .peek()
        .map_or(false, |token| !token.has_sub_tokens() && token.is_rooted())
    {
        // Include any rooting component boundary at the beginning of the token
        // sequence.
        prefix.push(MAIN_SEPARATOR);
    }
    // TODO: Replace `map`, `take_while`, and `flatten` with `map_while`
    //       when it stabilizes.
    prefix.push_str(
        &components(tokens)
            .map(|component| component.to_invariant_string().map(Cow::into_owned))
            .take_while(Option::is_some)
            .flatten()
            .join(&MAIN_SEPARATOR.to_string()),
    );
    if prefix.is_empty() {
        None
    } else {
        Some(prefix.into())
    }
}

pub(crate) fn invariant_prefix_upper_bound<A>(tokens: &[Token<A>]) -> usize {
    use crate::token::{
        TokenKind::{Separator, Wildcard},
        Wildcard::Tree,
    };

    let mut separator = None;
    for (n, token) in tokens.iter().map(Token::kind).enumerate() {
        match token {
            Separator => {
                separator = Some(n);
            },
            Wildcard(Tree { .. }) => {
                return n;
            },
            _ => {
                // TODO: This may be expensive.
                if token.to_invariant_string().is_some() {
                    continue;
                }
                return match separator {
                    Some(n) => n + 1,
                    None => 0,
                };
            },
        }
    }
    tokens.len()
}

pub(crate) fn parse(expression: &str) -> Result<Tokenized, ParseError> {
    use nom::{
        branch::alt,
        bytes::complete::{escaped_transform, is_not},
        character::complete::{digit1, none_of},
        combinator::{
            all_consuming,
            eof,
            map,
            map_opt,
            map_res,
            opt,
            peek,
            success,
            value,
            verify,
        },
        multi::{many0, many1},
        sequence::{delimited, preceded, separated_pair, terminated, tuple},
        IResult,
        Parser,
    };
    use supreme::{tag, ParserExt};

    use crate::token::FlagToggle::CaseInsensitive;

    #[derive(Clone, Copy, Debug, Error)]
    #[error("expected {subject}")]
    struct Expectation {
        subject: &'static str,
    }

    type ParseResult<'i, O> = IResult<Input<'i>, O, ErrorTree<'i>>;

    fn boe(input: Input) -> ParseResult<Input> {
        if input.state.subexpression == input.location() {
            Ok((input, input))
        } else {
            Err(ErrorMode::Error(ErrorTree::Base {
                location: input,
                kind:     BaseErrorKind::External(Box::new(Expectation {
                    subject: "beginning of expression",
                })),
            }))
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    const fn identity(input: Input) -> ParseResult<Input> {
        Ok((input, input))
    }

    fn flags<'i, F>(
        mut toggle: impl FnMut(FlagToggle) -> F,
    ) -> impl FnMut(Input<'i>) -> ParseResult<'i, ()>
    where
        F: Parser<Input<'i>, (), ErrorTree<'i>>,
    {
        move |input| {
            let (input, _) = many0(delimited(
                tag("(?"),
                many1(alt((
                    tuple((tag("i"), toggle(CaseInsensitive(true)))),
                    tuple((tag("-i"), toggle(CaseInsensitive(false)))),
                ))),
                tag(")"),
            ))(input)?;
            Ok((input, ()))
        }
    }

    // Explicit lifetimes prevent inference errors.
    #[allow(clippy::needless_lifetimes)]
    fn flags_with_state<'i>(input: Input<'i>) -> ParseResult<'i, ()> {
        flags(move |toggle| {
            move |mut input: Input<'i>| {
                match toggle {
                    CaseInsensitive(toggle) => {
                        input.state.flags.is_case_insensitive = toggle;
                    },
                }
                Ok((input, ()))
            }
        })(input)
    }

    // Explicit lifetimes prevent inference errors.
    #[allow(clippy::needless_lifetimes)]
    fn flags_without_state<'i>(input: Input<'i>) -> ParseResult<'i, ()> {
        flags(move |_| move |input: Input<'i>| Ok((input, ())))(input)
    }

    fn literal(input: Input) -> ParseResult<TokenKind<Annotation>> {
        map(
            verify(
                escaped_transform(
                    is_not("/?*$:<>()[]{},\\"),
                    '\\',
                    alt((
                        value("?", tag("?")),
                        value("*", tag("*")),
                        value("$", tag("$")),
                        value(":", tag(":")),
                        value("<", tag("<")),
                        value(">", tag(">")),
                        value("(", tag("(")),
                        value(")", tag(")")),
                        value("[", tag("[")),
                        value("]", tag("]")),
                        value("{", tag("{")),
                        value("}", tag("}")),
                        value(",", tag(",")),
                    )),
                ),
                |text: &str| !text.is_empty(),
            ),
            move |text| {
                TokenKind::Literal(Literal {
                    text:                text.into(),
                    is_case_insensitive: input.state.flags.is_case_insensitive,
                })
            },
        )(input)
    }

    fn separator(input: Input) -> ParseResult<TokenKind<Annotation>> {
        value(TokenKind::Separator, tag("/"))(input)
    }

    fn wildcard<'i>(
        terminator: impl Clone + Parser<Input<'i>, Input<'i>, ErrorTree<'i>>,
    ) -> impl FnMut(Input<'i>) -> ParseResult<'i, TokenKind<Annotation>> {
        alt((
            map(tag("?"), |_| TokenKind::from(Wildcard::One)).context("exactly-one"),
            map(
                tuple((
                    map(
                        alt((
                            tuple((value(true, tag("/")), flags_with_state)),
                            tuple((value(false, boe), flags_with_state)),
                        )),
                        |(prefix, _)| prefix,
                    )
                    .context("prefix"),
                    terminated(
                        tag("**"),
                        alt((
                            map(tuple((flags_with_state, tag("/"))), |(_, postfix)| postfix),
                            terminator.clone(),
                        ))
                        .context("postfix"),
                    ),
                )),
                |(is_rooted, _)| Wildcard::Tree { is_rooted }.into(),
            )
            .context("tree"),
            map(
                terminated(
                    tag("*"),
                    alt((
                        map(
                            peek(tuple((
                                flags_without_state,
                                is_not("*$").context("no terminating wildcard"),
                            ))),
                            |(_, right)| right,
                        ),
                        terminator.clone(),
                    )),
                ),
                |_| Wildcard::ZeroOrMore(Evaluation::Eager).into(),
            )
            .context("zero-or-more"),
            map(
                terminated(
                    tag("$"),
                    alt((
                        map(
                            peek(tuple((
                                flags_without_state,
                                is_not("*$").context("no terminating wildcard"),
                            ))),
                            |(_, right)| right,
                        ),
                        terminator,
                    )),
                ),
                |_| Wildcard::ZeroOrMore(Evaluation::Lazy).into(),
            )
            .context("zero-or-more"),
        ))
    }

    fn repetition(input: Input) -> ParseResult<TokenKind<Annotation>> {
        fn bounds(input: Input) -> ParseResult<(usize, Option<usize>)> {
            type BoundResult<T> = Result<T, <usize as FromStr>::Err>;

            alt((
                preceded(
                    tag(":"),
                    alt((
                        map_res(
                            separated_pair(digit1, tag(","), opt(digit1)),
                            |(lower, upper): (Input, Option<_>)| -> BoundResult<_> {
                                let lower = lower.parse::<usize>()?;
                                let upper =
                                    upper.map(|upper| upper.parse::<usize>()).transpose()?;
                                Ok((lower, upper))
                            },
                        )
                        .context("range"),
                        map_res(digit1, |n: Input| -> BoundResult<_> {
                            let n = n.parse::<usize>()?;
                            Ok((n, Some(n)))
                        })
                        .context("converged"),
                        success((1, None)),
                    )),
                ),
                success((0, None)),
            ))(input)
        }

        map_opt(
            delimited(
                tag("<"),
                tuple((
                    glob(move |input| peek(alt((tag(":"), tag(">"))))(input)).context("sub-glob"),
                    bounds.context("bounds"),
                )),
                tag(">"),
            ),
            |(tokens, (lower, upper))| match upper {
                Some(upper) => Repetition::new(tokens, lower..=upper).map(From::from),
                None => Repetition::new(tokens, lower..).map(From::from),
            },
        )(input)
    }

    fn class(input: Input) -> ParseResult<TokenKind<Annotation>> {
        fn archetypes(input: Input) -> ParseResult<Vec<Archetype>> {
            let escaped_character = |input| {
                alt((
                    none_of("[]-\\"),
                    alt((
                        value('[', tag("\\[")),
                        value(']', tag("\\]")),
                        value('-', tag("\\-")),
                    )),
                ))(input)
            };

            many1(alt((
                map(
                    separated_pair(escaped_character, tag("-"), escaped_character),
                    Archetype::from,
                ),
                map(escaped_character, Archetype::from),
            )))(input)
        }

        map(
            delimited(tag("["), tuple((opt(tag("!")), archetypes)), tag("]")),
            |(negation, archetypes)| Class { is_negated: negation.is_some(), archetypes }.into(),
        )(input)
    }

    fn alternative(input: Input) -> ParseResult<TokenKind<Annotation>> {
        preceded(
            tag("{"),
            map(
                supreme::many1(
                    glob(move |input| peek(alt((tag(","), tag("}"))))(input)).context("sub-glob"),
                    tag(","),
                    tag("}"),
                ),
                |alternatives: Vec<Vec<_>>| Alternative::from(alternatives).into(),
            ),
        )(input)
    }

    fn glob<'i>(
        terminator: impl 'i + Clone + Parser<Input<'i>, Input<'i>, ErrorTree<'i>>,
    ) -> impl Parser<Input<'i>, Vec<Token<'i, Annotation>>, ErrorTree<'i>> {
        #[cfg(any(feature = "diagnostics-inspect", feature = "diagnostics-report"))]
        fn annotate<'i, F>(
            parser: F,
        ) -> impl FnMut(Input<'i>) -> ParseResult<'i, Token<'i, Annotation>>
        where
            F: 'i + Parser<Input<'i>, TokenKind<'i, Annotation>, ErrorTree<'i>>,
        {
            map(pori::span(parser), |(span, kind)| Token::new(kind, span))
        }

        #[cfg(all(
            not(feature = "diagnostics-inspect"),
            not(feature = "diagnostics-report"),
        ))]
        fn annotate<'i, F>(
            parser: F,
        ) -> impl FnMut(Input<'i>) -> ParseResult<'i, Token<'i, Annotation>>
        where
            F: 'i + Parser<Input<'i>, TokenKind<'i, Annotation>, ErrorTree<'i>>,
        {
            map(parser, |kind| Token::new(kind, ()))
        }

        move |mut input: Input<'i>| {
            input.state.subexpression = input.location();
            supreme::many1(
                alt((
                    annotate(preceded(flags_with_state, literal)).context("literal"),
                    annotate(preceded(flags_with_state, repetition)).context("repetition"),
                    annotate(preceded(flags_with_state, alternative)).context("alternative"),
                    annotate(preceded(flags_with_state, wildcard(terminator.clone())))
                        .context("wildcard"),
                    annotate(preceded(flags_with_state, class)).context("class"),
                    annotate(preceded(flags_with_state, separator)).context("separator"),
                )),
                identity,
                terminator.clone(),
            )
            .parse(input)
        }
    }

    if expression.is_empty() {
        Ok(Tokenized { expression: expression.into(), tokens: vec![] })
    } else {
        // all_consuming(separated_pair(
        //     glob(eof),
        //     opt(tag("~")),
        //     opt(glob(eof)),
        // ));

        let input = Input::new(Expression::from(expression), ParserState::default());
        let tokens = all_consuming(glob(eof))(input)
            .map(|(_, tokens)| tokens)
            .map_err(|error| ParseError::new(expression, error))?;
        Ok(Tokenized { expression: expression.into(), tokens })
    }
}

fn fold_invariant_strings<'t, A, I>(tokens: I) -> Option<Cow<'t, str>>
where
    A: 't,
    I: IntoIterator<Item = &'t Token<'t, A>>,
    I::IntoIter: Clone,
{
    match tokens.into_iter().exactly_one() {
        Ok(token) => token.to_invariant_string(),
        Err(tokens) => tokens
            .fold(Some(String::new()), |output, token| {
                output.and_then(|mut output| {
                    token.to_invariant_string().map(move |string| {
                        output.push_str(string.as_ref());
                        output
                    })
                })
            })
            .map(From::from),
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use crate::token::{self, TokenKind};

    #[test]
    fn literal_case_insensitivity() {
        let tokenized = token::parse("(?-i)../foo/(?i)**/bar/**(?-i)/baz/*(?i)qux").unwrap();
        let literals: Vec<_> = tokenized
            .tokens()
            .iter()
            .filter_map(|token| match token.kind {
                TokenKind::Literal(ref literal) => Some(literal),
                _ => None,
            })
            .collect();

        assert!(!literals[0].is_case_insensitive); // `..`
        assert!(!literals[1].is_case_insensitive); // `foo`
        assert!(literals[2].is_case_insensitive); // `bar`
        assert!(!literals[3].is_case_insensitive); // `baz`
        assert!(literals[4].is_case_insensitive); // `qux`
    }

    #[test]
    fn invariant_prefix_path() {
        fn invariant_prefix_path(expression: &str) -> Option<PathBuf> {
            token::invariant_prefix_path(token::parse(expression).unwrap().tokens())
        }

        assert_eq!(invariant_prefix_path("/a/b").unwrap(), Path::new("/a/b"));
        assert_eq!(invariant_prefix_path("a/b").unwrap(), Path::new("a/b"));
        assert_eq!(invariant_prefix_path("a/*").unwrap(), Path::new("a/"));
        assert_eq!(invariant_prefix_path("a/*b").unwrap(), Path::new("a/"));
        assert_eq!(invariant_prefix_path("a/b*").unwrap(), Path::new("a/"));
        assert_eq!(invariant_prefix_path("a/b/*/c").unwrap(), Path::new("a/b/"));

        #[cfg_attr(not(any(unix, windows)), allow(unused))]
        let prefix = invariant_prefix_path("../foo/(?i)bar/(?-i)baz").unwrap();
        #[cfg(unix)]
        assert_eq!(prefix, Path::new("../foo"));
        #[cfg(windows)]
        assert_eq!(prefix, Path::new("../foo/bar"));

        assert!(invariant_prefix_path("**").is_none());
        assert!(invariant_prefix_path("a*").is_none());
        assert!(invariant_prefix_path("*/b").is_none());
        assert!(invariant_prefix_path("a?/b").is_none());
    }
}
