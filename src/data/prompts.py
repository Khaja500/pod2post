"""Prompts shared by the teacher (data generation), the student (training) and inference.

Stage A (map):    one transcript chunk -> episode notes (JSON)
Stage B (reduce): merged notes + target keywords -> article (Markdown) + social pack (JSON)
The system prompt alone tells the single LoRA adapter which task it is doing (spec §3).
"""
from __future__ import annotations

STAGE_A_SYSTEM = (
    "You are an editorial research assistant. You read one segment of a podcast or interview "
    "transcript and extract structured notes for a writer. Use only what the segment says: never add "
    "facts, names or numbers that are not in the text, and copy quotes word for word. "
    "Reply with a single JSON object and nothing else."
)

STAGE_A_USER = """Episode: {title}
Segment {index} of {total}.

Transcript segment:
<<<
{chunk}
>>>

Return a JSON object with exactly these keys:
- "summary": 2-3 sentences on what this segment covers.
- "key_points": 3-8 short strings with the substantive ideas, arguments or explanations.
- "claims": factual claims made by speakers, as objects {{"speaker", "claim", "timestamp"}}.
- "quotes": 1-3 memorable verbatim quotes of at most 40 words, as objects {{"speaker", "quote", "timestamp"}}.
- "entities": people, organisations, products and places named in the segment.
- "numbers": figures, dates or statistics, each with a few words of context.
- "keywords": 2-5 search phrases a reader might use to find this topic.
Write every name exactly as it appears in the transcript; never expand or complete it (if the transcript \
says "Trump", write "Trump"). Each quote must be one continuous passage from a single speaker's line, copied \
exactly; never join text from different lines. Use the timestamp shown before a line ("mm:ss" or "h:mm:ss") \
or null. Use [] or "" when the segment has nothing for a key."""

STAGE_B_SYSTEM = (
    "You are an SEO content writer for podcasts. From an episode's notes you write a long-form article "
    "and a social media pack. Everything you write must be supported by the notes: do not invent facts, "
    "names, numbers or quotes. Follow the output format exactly."
)

STAGE_B_USER = """Episode: {title}
Primary keyword: {primary_keyword}
Secondary keywords: {secondary_keywords}

Episode notes:
<<<
{notes}
>>>

Write the article and the social pack in exactly this format:

<article>
# <title: at most 60 characters, contains the primary keyword>
Meta description: <one sentence, at most 160 characters, contains the primary keyword>

<introduction that uses the primary keyword within its first 100 words>

## <at least three H2 sections; one H2 contains the primary keyword; use ### subsections where useful>

## FAQ
### <3-5 questions a listener might ask, each answered in 2-3 sentences>
</article>
<social>
{{"linkedin": "<post of 80-150 words>", "x_thread": ["<4-6 posts, each at most 280 characters>"], \
"newsletter": "<blurb of 40-70 words>", "pull_quotes": ["<3-5 quotes copied word for word from the notes>"]}}
</social>

The article body is 700-1000 words. Use the exact primary keyword 5-10 times in total, never more, and the \
secondary keywords where they fit. The <social> block is one single JSON object containing all four keys."""


def stage_a_messages(title: str, chunk: str, index: int, total: int) -> list[dict]:
    return [{"role": "system", "content": STAGE_A_SYSTEM},
            {"role": "user", "content": STAGE_A_USER.format(title=title, chunk=chunk, index=index, total=total)}]


def stage_b_messages(title: str, notes: str, primary_keyword: str, secondary_keywords: list[str]) -> list[dict]:
    user = STAGE_B_USER.format(title=title, notes=notes, primary_keyword=primary_keyword,
                               secondary_keywords=", ".join(secondary_keywords) or "none")
    return [{"role": "system", "content": STAGE_B_SYSTEM}, {"role": "user", "content": user}]
