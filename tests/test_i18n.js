/* Optional JS checks: node tests/test_i18n.js; no third-party dependencies. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const translations = JSON.parse(fs.readFileSync(path.join(root, 'static/translations.json'), 'utf8'));
const helper = fs.readFileSync(path.join(root, 'static/i18n.js'), 'utf8');
const script = fs.readFileSync(path.join(root, 'static/script.js'), 'utf8');
new vm.Script(script); // Modern browser script must parse.
for (const match of script.matchAll(/\bt\(("(?:[^"\\]|\\.)*")/g)) {
    const key = JSON.parse(match[1]);
    if (/[\u3400-\u9fff]/.test(key)) assert.ok(Object.hasOwn(translations, key), key);
}
for (const language of ['en', 'zh-CN']) {
    const storage = new Map();
    const context = {
        GO_LANGUAGE: language, GO_TRANSLATIONS: translations,
        URL, console: {warn() {}}, document: {cookie: ''},
        location: {href: 'http://localhost:5053/?lang=' + language, assign(url) { this.assigned = url; }},
        sessionStorage: {
            getItem: key => storage.get(key) || null,
            setItem: (key, value) => storage.set(key, value),
            removeItem: key => storage.delete(key),
        },
    };
    context.window = context;
    vm.createContext(context);
    vm.runInContext(helper, context);
    const api = context.GoI18n;
    assert.equal(api.t('__proto__'), '__proto__');
    assert.equal(api.t('constructor'), 'constructor');
    assert.equal(api.t('{0}{1}（行 {2}，列 {3}）', ['D', 16, 3, 3]),
                 language === 'en' ? 'D16 (row 3, column 3)' : 'D16（行 3，列 3）');
    assert.equal(api.t('强化学习 Epoch 4（770dfc61）'),
                 language === 'en' ? 'RL Epoch 4 (770dfc61)' : '强化学习 Epoch 4（770dfc61）');
    assert.equal(api.t('棋谱结果：{0}', ['literal$&{9}']),
                 (language === 'en' ? 'SGF result: ' : '棋谱结果：') + 'literal$&{9}');
    if (language === 'en') assert.doesNotMatch(api.error('未知服务器错误'), /[\u3400-\u9fff]/);
    const state = {blackModel: 'rl-epoch-kept', whiteModel: 'rl', komi: '7.5', sgfLoaded: true};
    api.changeLanguage(language === 'en' ? 'zh-CN' : 'en', state);
    assert.ok(context.document.cookie.includes('go_language='));
    assert.deepEqual(JSON.parse(JSON.stringify(api.takeResume())), state);
    assert.equal(api.takeResume(), null);
}
console.log('Bilingual JS checks passed: catalog coverage, labels, errors, interpolation, state persistence.');
