# Personal writing recovery audit

This review queue is excluded from the generated site. The Korean originals are the source of truth; translation must not silently rewrite them.

## Publication rule

A personal detail is publishable when it would be reasonable to disclose to someone at the beginning of a casual relationship.

- Publishable: poverty, debt, class background, career insecurity, ordinary regret or inferiority, past dating lessons, rough language, and controversial political, gender, or regional opinions.
- Keep private: mental-health diagnoses, medication, suicidal thoughts or attempts, and writing that clearly exposes an acute mental-health crisis.
- Manual review only: identifiable details about another person, highly intimate relationship history, or an unfinished fragment. These are not rejected for tone or ideology.

## Restored as Korean originals

- `2017-11-26-welcome-journal.markdown`
- `2018-06-20-productivity_tools_i_use.md.md`
- `2019-10-19-journal.md`
- `2019-7-6-journal.md`
- `2020-12-20-karyuushikou.md`
- `2021-03-19-nikki.md`
- `2021-06-13-thoughts.md`
- `2022-08-10-nikki.md`

## Restored with four languages

- `2022-10-11-nikki.md`
- `2023-02-20-paper-reflections.md`, recovered from Git history
- `2021-12-09-thoughts.md`
- `2022-02-11-uso.md`
- `2025-01-09-nikki.md`
- `2017-12-13-오랜만에-일기.markdown`
- `2018-03-27-일기.md`
- `2019-8-27-journal.md`

The Korean sections in all files preserve the originals. Japanese, Chinese, and English are translations rather than rewrites of the Korean source.

## Keep private under the mental-health rule

- `2018-06-25-일기.md`: recurring depression is explicit.
- `2020-01-11-journal.md`: bipolar diagnosis is explicit.
- `2020-08-24-journal.md`: contains a strong statement questioning whether the author should be alive.
- `2021-10-23-thoughts.md`: dominated by severe pessimism and self-negation.
- `2022-10-26-nikki.md`: suicidal behavior and intent are explicit.
- `2022-11-03-nikki.md`: bipolar disorder and mental illness are explicit.
- `2025-01-12-journal.md`: reads as an acute mental-health crisis.
- `_drafts/Untitled-2.md`: depression is explicit.
- deleted `2023-01-19-19일의 일기.md`: depression, medication, and concealment are explicit.

## Manual review, not rejected

- `2018-03-25-일기.md`: unusually strong self-attack and relationship self-disclosure.
- `2018-03-26-흙수저로서.md`: safe class disclosure, but only a one-sentence fragment.
- `2021-02-11-skku-amusement-park.md`: severe self-attack without an explicit diagnosis.
- `2022-10-25-nikki.md`: incomplete.
- `_drafts/2022-08-16-thoughts.md`: highly intimate sexual opinions; not a mental-health exclusion.
- `_drafts/be-not-disclosed.md`: contains slurs and is unfinished; not an ideological exclusion.
- deleted `2023-01-29-1월 29일의 일기.md`: detailed memories of former partners.
- deleted `2023-05-13-13일의 일기.md`: too fragmentary to publish as a post.

## Already public or duplicated elsewhere

- `2018-6-12-일기.md`
- `2019-7-23-get-a-job.md`
- `2021-12-27-learn-something-in-programmer-way.md`
- `2022-12-08-nikki.md`
- `2023-03-01-change.md`

## Privacy note

The `일기_저장소` directory is excluded from the generated site. Its files remain visible in the Git repository itself if the repository is public. Source-level privacy requires moving sensitive originals to private storage and handling Git history separately.

## Multilingual authoring format

One post file produces one URL. The reader's saved language is used first, then a supported browser language, then `default_lang`.

```yaml
titles:
  ko: "한국어 제목"
  ja: "日本語のタイトル"
  zh: "中文标题"
  en: "English title"
languages: [ko, ja, zh, en]
default_lang: ko
```

```html
<section class="post-translation" data-lang="ko" data-default-language markdown="1">
한국어 원문
</section>

<section class="post-translation" data-lang="ja" markdown="1">
日本語訳
</section>
```
