import assert from 'node:assert/strict';
import { inspectSource, validateTranslation } from './translation-source.mjs';

const korean = '한글 원문은 절대로 수정하면 안 되는 내용이다.\n\n# 원문 제목\n\n`code` **강조**';
const raw = `---\ntitle: 원문\n---\n${korean}\n`;
const original = inspectSource(raw);
assert.equal(original.language, 'ko');
assert.equal(inspectSource('---\ntitle: English\n---\nEnglish text.').language, 'en');
assert.throws(() => inspectSource('No front matter'), /front matter/);

const section = (lang, body) => `<section class="post-translation" data-lang="${lang}"${lang === 'ko' ? ' data-default-language' : ''} markdown="1">\n${body}\n</section>`;
const translation = `---\ntitle: 원문\nlanguages: [ko, ja, zh, en]\ndefault_lang: ko\n---\n${['ko', 'ja', 'zh', 'en'].map(lang => section(lang, lang === 'ko' ? korean : 'Translated text.')).join('\n\n')}`;
validateTranslation(translation, 'ko', original.hash);
validateTranslation(translation.replaceAll('\n', '\r\n'), 'ko', original.hash);
assert.deepEqual(inspectSource(translation), original);
assert.throws(() => validateTranslation(translation.replace('한글 원문', '수정된 원문'), 'ko', original.hash), /changed the source/);
assert.throws(() => validateTranslation(translation + '\n' + section('ja', 'Duplicate'), 'ko', original.hash), /exactly one/);
assert.throws(() => validateTranslation(translation.replace(' data-default-language', ''), 'ko', original.hash), /opening tag/);
assert.throws(() => validateTranslation(translation.replace('languages: [ko, ja, zh, en]', 'languages: [ko, en]'), 'ko', original.hash), /front matter/);
assert.throws(() => validateTranslation(translation.replace('default_lang: ko', 'default_lang: en'), 'ko', original.hash), /front matter/);
assert.notEqual(inspectSource(raw.replace(korean, '\u00a0' + korean)).hash, original.hash, 'Unicode whitespace must not be silently stripped');
console.log('Translation source protection: hash parity, CRLF, unchanged source, and rejection checks passed.');
