/* Shared text catalog supplied by Flask. Language changes issue GETs only. */
window.GoI18n = (() => {
    const language = window.GO_LANGUAGE === 'en' ? 'en' : 'zh-CN';
    const catalog = window.GO_TRANSLATIONS || {};
    const reverse = new Map(Object.entries(catalog).map(([zh, en]) => [en, zh]));
    function t(message, values = []) {
        if (typeof message !== 'string') return message;
        let source = reverse.get(message) || message;
        let text = language === 'en' && Object.hasOwn(catalog, source) ? catalog[source] : source;
        if (language === 'en') {
            text = text.replace(/^强化学习 Epoch (\d+)（([^）]+)）$/, 'RL Epoch $1 ($2)')
                       .replace(/^历史模型（([^）]+)）$/, 'Historical policy ($1)');
        } else {
            text = text.replace(/^RL Epoch (\d+) \(([^)]+)\)$/, '强化学习 Epoch $1（$2）')
                       .replace(/^Historical policy \(([^)]+)\)$/, '历史模型（$1）');
        }
        // Callback substitution preserves literal $, braces and filenames in values.
        return text.replace(/\{(\d+)\}/g, (whole, index) => index < values.length ? String(t(values[index])) : whole);
    }
    function error(message) {
        if (typeof message !== 'string') return message;
        const translated = t(message);
        if (language === 'en' && /[\u3400-\u9fff]/.test(translated)) {
            console.warn('Server error:', message);
            return t('请求失败，请查看服务器日志。');
        }
        return translated;
    }
    function takeResume() {
        try {
            const raw = sessionStorage.getItem('go-language-resume');
            sessionStorage.removeItem('go-language-resume');
            return raw ? JSON.parse(raw) : null;
        } catch { return null; }
    }
    function changeLanguage(next, state) {
        if (!['en', 'zh-CN'].includes(next) || next === language) return;
        try { sessionStorage.setItem('go-language-resume', JSON.stringify(state)); } catch {}
        document.cookie = `go_language=${next}; Path=/; Max-Age=31536000; SameSite=Lax`;
        const url = new URL(location.href);
        url.searchParams.set('lang', next);
        location.assign(url.href);
    }
    return {language, t, error, takeResume, changeLanguage};
})();
const t = window.GoI18n.t;
const localizeError = window.GoI18n.error;
