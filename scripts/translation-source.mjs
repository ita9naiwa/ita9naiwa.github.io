import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const sectionPattern = /<section\b[^>]*data-lang="(ko|ja|zh|en)"[^>]*>[\t\n\v\f\r ]*\n([\s\S]*?)\n<\/section>/g;
const languages = ['ko', 'ja', 'zh', 'en'];
// Match the previous Ruby String#strip exactly, including NUL, not Unicode whitespace.
const sourceHash = source => createHash('sha256').update(source.replace(/\r\n/g, '\n').replace(/^[\x00\x09-\x0d\x20]+|[\x00\x09-\x0d\x20]+$/g, '')).digest('hex');

export function inspectSource(text) {
  const korean = [...text.matchAll(sectionPattern)].find(section => section[1] === 'ko');
  if (korean) return { language: 'ko', hash: sourceHash(korean[2]) };
  const front = text.match(/^---[\t\n\v\f\r ]*\n[\s\S]*?\n---[\t\n\v\f\r ]*\n/);
  if (!front) throw new Error('Markdown front matter was not found.');
  const source = text.slice(front[0].length);
  return { language: (source.match(/[가-힣]/g) || []).length >= 10 ? 'ko' : 'en', hash: sourceHash(source) };
}

export function validateTranslation(text, language, originalHash) {
  const sections = [...text.matchAll(sectionPattern)];
  if (!languages.every(lang => sections.filter(section => section[1] === lang).length === 1)) {
    throw new Error('Translation rejected: expected exactly one ko, ja, zh, and en section.');
  }
  for (const lang of languages) {
    const opening = `<section class="post-translation" data-lang="${lang}"${lang === language ? ' data-default-language' : ''} markdown="1">`;
    if (text.split(opening).length !== 2) throw new Error(`Translation rejected: invalid opening tag for ${lang}.`);
  }
  const source = sections.find(section => section[1] === language);
  if (!source || sourceHash(source[2]) !== originalHash) {
    throw new Error('Translation rejected: Codex changed the source text. The original post was left untouched.');
  }
  if (!/^languages:[\t\n\v\f\r ]*\[ko, ja, zh, en\][\t\n\v\f\r ]*$/m.test(text) || !new RegExp(`^default_lang:[\\t\\n\\v\\f\\r ]*${language}[\\t\\n\\v\\f\\r ]*$`, 'm').test(text)) {
    throw new Error('Translation rejected: multilingual front matter is incomplete.');
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    const [command, ...args] = process.argv.slice(2);
    if (command === 'inspect' && args.length === 1) {
      const { language, hash } = inspectSource(readFileSync(args[0], 'utf8'));
      console.log(`${language}\n${hash}`);
    } else if (command === 'validate' && args.length === 3) {
      validateTranslation(readFileSync(args[2], 'utf8'), args[0], args[1]);
    } else throw new Error('Usage: translation-source.mjs inspect FILE | validate LANGUAGE HASH FILE');
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
