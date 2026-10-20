# Data sources

Licenses checked 2026-10-20 against each source's current card or README. Everything here is used for a non-commercial portfolio project. Processed data lives only in the **private** Kaggle dataset `khajaaflal/pod2post-data`.

| Source | Use | License (verified) | Taken | Notes |
|---|---|---|---|---|
| **MediaSum** (`ccdv/mediasum` mirror of github.com/zcgzcgzcg1/MediaSum) | Training transcripts (Stage A/B inputs) | README: "restrict your usage of this dataset to research purpose only". The GitHub repo has no license file | 2,000 interviews from `val_data.zip`: 1,775 sampled at 600–6,000 words, plus all 225 at 6,000–25,000 words. Seed 42 | NPR/CNN transcripts. Cite Zhu et al. 2021 (arXiv:2103.06410) |
| **TalkRL podcast** (HF `RamAnanth1/talkrl-podcast`) | Held-out test (38 episodes) | **No license stated** on the card | All rows except the 305-word "About TalkRL" trailer | Whisper (base) transcripts with timestamps, no speakers. Evaluation only; never redistributed |
| **The Changelog** (github.com/thechangelog/transcripts, `podcast/`) | Held-out test (12 episodes) | **CC BY-SA 4.0** (repo LICENSE) | Episodes #662–673, the 12 most recent | Human-edited transcripts with speakers. The HF copies by `willtheorangeguy` are tagged MIT, but they derive from this repo, so CC BY-SA applies. Attribute "The Changelog" if any excerpt is published |
| **SPoRC** (HF `blitt/SPoRC`) | Podcast-style training transcripts | Card license "other"; access is gated (accept terms while logged in) | **Not used yet** | Needs a Hugging Face login, which this machine doesn't have. Optional: it would add real Whisper-style podcast transcripts to training |
| **Creative-Commons audio** (Internet Archive) | End-to-end Whisper test (Phase 2) | CC (variant per show) | — | To be picked in Phase 2 |
| ~~Spotify Podcast Dataset~~ | — | Discontinued Dec 2023 | — | Do not use |

## Test set

There are 50 episodes: 38 TalkRL and 12 Changelog, frozen on 2026-10-20 **before any generation or training**. Their IDs are stored as SHA-256 hashes in `data/processed/test_ids.sha256`, which is git-tracked and also shipped in the Kaggle dataset. `src/data/testset.assert_no_test_leak()` guards every training entry point. Re-running `src.data.download` refuses to change the test selection.

| Split | Episodes | Words (median / min / max) | Chunks (≤1,500 tokens, 150 overlap) |
|---|---|---|---|
| train (MediaSum) | 2,000 | 1,504 / 600 / 17,304 | 5,585 (median 2 per episode, max 19) |
| test | 50 | 10,792 / 3,751 / 21,968 | 630 (median 11 per episode, max 28) |
