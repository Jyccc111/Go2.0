/* Presentation pacing only. No changes to policy sampling or game rules. */
window.GoPlayback = {
    delayMs(enabled, value) {
        const delay = Number(value);
        return enabled && [500, 1000, 2000, 3000].includes(delay) ? delay : 0;
    },
    async wait(enabled, value) {
        const delay = this.delayMs(enabled, value);
        if (delay) await new Promise(resolve => setTimeout(resolve, delay));
    },
};
