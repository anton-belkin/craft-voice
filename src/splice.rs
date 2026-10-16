//! What the user said, with what they pointed at while saying it.
//!
//! While the user speaks they click cells, paragraphs or shapes. Each click is a [`Pointer`]: the
//! moment (seconds since the recording started) and an app-defined anchor. [`splice`] puts a marker
//! `⟦n⟧` into the transcript where the user was in their speech at that moment: inside a segment
//! of the transcript by interpolating its time span over its characters, then rounded to the
//! nearest word boundary. Pointing is not interpreted: no matching of "this" or "это"; whoever
//! reads the text (an agent) resolves the references from context.

/// One stretch of speech with its time span, in seconds from the start of the recording.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Segment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

/// A transcription: the text, its segments and the language that was heard.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub segments: Vec<Segment>,
    /// ISO 639-1 code (`en`, `ru`…), empty when unknown.
    pub language: String,
}

impl Transcript {
    /// A transcript made of segments (the text is their words, single-spaced).
    pub fn from_segments(segments: Vec<Segment>, language: impl Into<String>) -> Self {
        let text = segments.iter().flat_map(|s| s.text.split_whitespace()).collect::<Vec<_>>().join(" ");
        Self { text, segments, language: language.into() }
    }

    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }
}

/// The user pointed at `anchor` `at` seconds into the recording.
#[derive(Clone, Debug, PartialEq)]
pub struct Pointer<A> {
    pub at: f32,
    pub anchor: A,
}

/// The transcript with markers, and what each marker stands for (numbered from 1, in the order
/// the user pointed; pointing at the same anchor again reuses its number).
#[derive(Clone, Debug, PartialEq)]
pub struct Spliced<A> {
    pub text: String,
    pub refs: Vec<(u32, A)>,
}

/// The marker for reference `n`: `⟦n⟧`.
pub fn marker(n: u32) -> String {
    format!("⟦{n}⟧")
}

/// The numbers of the markers in a text, in order (`"a ⟦2⟧ b ⟦1⟧"` → `[2, 1]`).
pub fn markers_in(text: &str) -> Vec<u32> {
    let mut out = vec![];
    let mut rest = text;
    while let Some(i) = rest.find('⟦') {
        let after = &rest[i + '⟦'.len_utf8()..];
        match after.find('⟧') {
            Some(j) => {
                if let Ok(n) = after[..j].trim().parse::<u32>() {
                    out.push(n);
                }
                rest = &after[j + '⟧'.len_utf8()..];
            }
            None => break,
        }
    }
    out
}

/// Puts a marker for each pointer into the transcript (see the module docs).
pub fn splice<A: PartialEq + Clone>(t: &Transcript, pointers: &[Pointer<A>]) -> Spliced<A> {
    // Words of every segment, and the word range each segment covers.
    let mut words: Vec<&str> = vec![];
    let mut spans: Vec<(f32, f32, usize, usize)> = vec![];
    for s in &t.segments {
        let w0 = words.len();
        words.extend(s.text.split_whitespace());
        let w1 = words.len();
        if w1 > w0 && s.start.is_finite() && s.end.is_finite() {
            spans.push((s.start, s.end.max(s.start), w0, w1));
        }
    }
    // A transcript without segments (or timing) still takes markers: at its end.
    if spans.is_empty() && words.is_empty() {
        words.extend(t.text.split_whitespace());
    }
    let n_words = words.len();

    // Number the references in the order the user pointed.
    let mut order: Vec<&Pointer<A>> = pointers.iter().collect();
    order.sort_by(|a, b| a.at.total_cmp(&b.at));
    let mut refs: Vec<(u32, A)> = vec![];
    // (boundary, n) in pointing order.
    let mut at: Vec<(usize, u32)> = vec![];
    for p in order {
        let n = match refs.iter().find(|(_, a)| *a == p.anchor) {
            Some((n, _)) => *n,
            None => {
                let n = u32::try_from(refs.len()).unwrap_or(u32::MAX).saturating_add(1);
                refs.push((n, p.anchor.clone()));
                n
            }
        };
        let b = if spans.is_empty() { n_words } else { boundary(&words, &spans, p.at) };
        if !at.contains(&(b, n)) {
            at.push((b, n));
        }
    }

    let mut out: Vec<String> = vec![];
    for k in 0..=n_words {
        for (_, n) in at.iter().filter(|(b, _)| *b == k) {
            out.push(marker(*n));
        }
        if let Some(w) = words.get(k) {
            out.push((*w).to_string());
        }
    }
    Spliced { text: out.join(" "), refs }
}

/// The word boundary (0 = before the first word, `words.len()` = after the last) for a moment.
fn boundary(words: &[&str], spans: &[(f32, f32, usize, usize)], t: f32) -> usize {
    let (Some(first), Some(last)) = (spans.first(), spans.last()) else { return words.len() };
    if !t.is_finite() || t <= first.0 {
        return first.2;
    }
    if t >= last.1 {
        return last.3;
    }
    for (i, &(start, end, w0, w1)) in spans.iter().enumerate() {
        if t >= start && t <= end {
            return within(words, w0, w1, if end > start { (t - start) / (end - start) } else { 1.0 });
        }
        // In the silence before the next segment: the nearer end of the speech around it.
        if let Some(next) = spans.get(i + 1)
            && t > end
            && t < next.0
        {
            return if t - end <= next.0 - t { w1 } else { next.2 };
        }
    }
    last.3
}

/// The word boundary in words `w0..w1` nearest to a fraction of their characters.
fn within(words: &[&str], w0: usize, w1: usize, frac: f32) -> usize {
    let lens: Vec<usize> = words.get(w0..w1).unwrap_or_default().iter().map(|w| w.chars().count()).collect();
    let total = lens.iter().sum::<usize>() + lens.len().saturating_sub(1);
    let target = frac.clamp(0.0, 1.0) * total as f32;
    // Boundary j sits before word j: at the start, in the middle of the space before word j, or
    // at the end. The first of two equally near boundaries wins.
    let mut best = (w0, f32::INFINITY);
    let mut chars = 0usize;
    for j in 0..=lens.len() {
        let pos = match j {
            0 => 0.0,
            j if j == lens.len() => total as f32,
            j => chars as f32 + (j - 1) as f32 + 0.5,
        };
        let d = (pos - target).abs();
        if d < best.1 {
            best = (w0 + j, d);
        }
        chars += lens.get(j).copied().unwrap_or(0);
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f32, end: f32, text: &str) -> Segment {
        Segment { start, end, text: text.into() }
    }

    fn p(at: f32, a: &str) -> Pointer<String> {
        Pointer { at, anchor: a.to_string() }
    }

    #[test]
    fn marker_inside_a_segment_lands_on_a_word_boundary() {
        // 19 characters over 2 seconds; boundaries at 0, 4.5, 7.5, 12.5 and 19.
        let t = Transcript::from_segments(vec![seg(0.0, 2.0, " what is this number")], "en");
        assert_eq!(t.text, "what is this number");
        // 0.75 s → character 7.1 → the space between "is" and "this".
        let s = splice(&t, &[p(0.75, "C16")]);
        assert_eq!(s.text, "what is ⟦1⟧ this number");
        assert_eq!(s.refs, vec![(1, "C16".to_string())]);
        // 0.2 s → character 1.9 → nearer the start than the space after "what".
        assert_eq!(splice(&t, &[p(0.2, "A1")]).text, "⟦1⟧ what is this number");
        assert_eq!(splice(&t, &[p(0.35, "A1")]).text, "what ⟦1⟧ is this number");
        // The very end of the segment: after the last word.
        assert_eq!(splice(&t, &[p(2.0, "A1")]).text, "what is this number ⟦1⟧");
    }

    #[test]
    fn clicks_without_speech_go_to_the_nearest_end() {
        let t = Transcript::from_segments(vec![seg(1.0, 2.0, "revenue here"), seg(4.0, 5.0, "and costs there")], "en");
        assert_eq!(splice(&t, &[p(0.1, "X")]).text, "⟦1⟧ revenue here and costs there");
        assert_eq!(splice(&t, &[p(9.0, "X")]).text, "revenue here and costs there ⟦1⟧");
        // In the gap, nearer the first segment's end, then nearer the second's start: the same
        // boundary between the two (they're consecutive words).
        assert_eq!(splice(&t, &[p(2.4, "X")]).text, "revenue here ⟦1⟧ and costs there");
        assert_eq!(splice(&t, &[p(3.6, "X")]).text, "revenue here ⟦1⟧ and costs there");
    }

    #[test]
    fn numbers_follow_the_pointing_order_and_repeat_for_the_same_anchor() {
        let t = Transcript::from_segments(vec![seg(0.0, 4.0, "compare this with this and this again")], "en");
        let s = splice(&t, &[p(3.9, "B2"), p(0.9, "B2"), p(2.0, "C3")]);
        assert_eq!(s.refs, vec![(1, "B2".to_string()), (2, "C3".to_string())]);
        assert_eq!(markers_in(&s.text), vec![1, 2, 1]);
    }

    #[test]
    fn several_markers_at_one_place_keep_their_order() {
        let t = Transcript::from_segments(vec![seg(0.0, 1.0, "these")], "en");
        let s = splice(&t, &[p(1.5, "A"), p(1.6, "B")]);
        assert_eq!(s.text, "these ⟦1⟧ ⟦2⟧");
    }

    #[test]
    fn russian_text_is_counted_in_characters_not_bytes() {
        // "что это за цифра" — 16 characters over 1.6 s.
        let t = Transcript::from_segments(vec![seg(0.0, 1.6, "что это за цифра")], "ru");
        // 0.8 s → character 8 → nearest the space before "за" (7.5).
        assert_eq!(splice(&t, &[p(0.8, "C16")]).text, "что это ⟦1⟧ за цифра");
    }

    #[test]
    fn empty_and_odd_input_never_panics() {
        let empty = Transcript::default();
        assert_eq!(splice(&empty, &[p(0.5, "A1")]).text, "⟦1⟧");
        assert_eq!(splice::<String>(&empty, &[]).text, "");
        let t = Transcript::from_segments(vec![seg(f32::NAN, 1.0, "x"), seg(2.0, 1.0, "y z"), seg(3.0, 3.0, "")], "");
        let s = splice(&t, &[p(f32::NAN, "a"), p(f32::INFINITY, "b"), p(-3.0, "c"), p(2.0, "d")]);
        assert_eq!(markers_in(&s.text).len(), 4);
        // No segments, only text: markers at the end.
        let t = Transcript { text: "hello there".into(), ..Default::default() };
        assert_eq!(splice(&t, &[p(0.0, "a")]).text, "hello there ⟦1⟧");
    }

    #[test]
    fn markers_are_found_again() {
        assert_eq!(markers_in("a ⟦12⟧ b ⟦x⟧ ⟦3⟧ ⟦"), vec![12, 3]);
    }
}
