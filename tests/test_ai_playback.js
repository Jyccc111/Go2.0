/* node tests/test_ai_playback.js — virtual timers; no model/network dependencies. */
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const helper = fs.readFileSync(path.join(root, 'static/ai_playback.js'), 'utf8');
const script = fs.readFileSync(path.join(root, 'static/script.js'), 'utf8');
const start = script.indexOf('async function runSelfPlay(');
const end = script.indexOf('\nasync function playMove(', start);
assert.ok(start >= 0 && end > start);
new vm.Script(script);

async function play({enabled = true, delay = '1000', terminal = false, error = false, change} = {}) {
    const events = [], timers = [];
    let requests = 0, renders = 0;
    const context = {
        window: {}, slowAiPlayback: {checked: enabled}, aiMoveDelay: {value: delay},
        isSelfPlaying: false, currentColor: 1, gameStatus: {textContent: ''},
        t: text => text, localizeError: text => text,
        requestAnimationFrame: callback => callback(),
        setTimeout: (callback, ms) => { timers.push(ms); events.push('wait'); callback(); },
        fetch: async () => {
            requests += 1; events.push('request');
            return {ok: !error, json: async () => ({success: !error, error: 'failure',
                game_over: requests === 3, result: requests === 3 ? 'W+R' : null,
                next_player: requests % 2 ? -1 : 1})};
        },
        clearAiRecommendations() {}, showSessionState() {},
        drawBoardState() {
            renders += 1; events.push('draw');
            if (renders === 1 && change) change(context);
        },
    };
    vm.createContext(context);
    vm.runInContext(helper + '\n' + script.slice(start, end), context);
    assert.equal(context.window.GoPlayback.delayMs(true, '-500'), 0);
    assert.equal(context.window.GoPlayback.delayMs(true, 'invalid'), 0);
    assert.equal(context.window.GoPlayback.delayMs(true, '9999999'), 0);
    await context.runSelfPlay({game_over: terminal});
    assert.equal(context.isSelfPlaying, false);
    return {events, timers, requests, renders};
}

module.exports = (async () => {
    const slow = await play();
    assert.deepEqual(slow.timers, [1000,1000,1000]);
    assert.deepEqual(slow.events, ['wait','request','draw','wait','request','draw','wait','request','draw']);
    assert.equal(slow.requests, 3); // Resignation/terminal response schedules no extra move.
    const fast = await play({enabled:false});
    assert.deepEqual(fast.timers, []);
    assert.equal(fast.requests, slow.requests);
    assert.equal(fast.renders, slow.renders);
    const changed = await play({delay:'3000', change: ctx => {ctx.aiMoveDelay.value = '500';}});
    assert.deepEqual(changed.timers, [3000,500,500]);
    const disabled = await play({change: ctx => {ctx.slowAiPlayback.checked = false;}});
    assert.deepEqual(disabled.timers, [1000]);
    const ended = await play({terminal:true});
    assert.equal(ended.requests, 0);
    assert.deepEqual(ended.timers, []);
    const failure = await play({error:true});
    assert.equal(failure.requests, 1);
    assert.equal(failure.renders, 0);
    console.log('AI playback checks passed: delays, fast mode, live changes, terminal stop and errors.');
})();
