const canvas = document.getElementById("go-board");
const ctx = canvas.getContext("2d");
const gridWidth = canvas.width/20;
const toggleAirButton = document.getElementById("toggle-air");
const passMoveButton = document.getElementById("pass-move");
const startGameButton = document.getElementById("start-game");
const showRecommendationsButton = document.getElementById("show-ai-recommendations");
const recommendationsText = document.getElementById("ai-recommendations");
const predictGameResultButton = document.getElementById("predict-game-result");
const valuePredictionText = document.getElementById("value-prediction");
const countScoreButton = document.getElementById("count-score");
const countSgfScoreButton = document.getElementById("count-sgf-score");
let scoringReplay = false;
const scoreResult = document.getElementById("score-result");
let scoreRequestId = 0;
let scoreTerritory = null;
let scoredBoard = null;
const gameStatus = document.getElementById("game-status");
const sessionDialog = document.getElementById("session-dialog");
const policyModelBlackSelect = document.getElementById("policy-model-black");
const policyModelWhiteSelect = document.getElementById("policy-model-white");
const policyModelStatus = document.getElementById("policy-model-status");
const refreshPolicyModelsButton = document.getElementById("refresh-policy-models");
const policyCatalogStatus = document.getElementById("policy-catalog-status");
let policyCatalogLoading = false;
const sessionInitMessage = document.getElementById("session-init-message");
let sessionInitializing = false;
const sgfDialog = document.getElementById("sgf-dialog");
document.getElementById("open-session-dialog").addEventListener("click", () => {
    if (!sessionInitializing) {
        sessionInitMessage.textContent = "";
        sessionInitMessage.dataset.state = "";
    }
    sessionDialog.showModal();
    loadPolicyModels();
});
sessionDialog.addEventListener("cancel", event => {
    if (sessionInitializing) event.preventDefault();
});
document.getElementById("open-sgf-dialog").addEventListener("click", () => sgfDialog.showModal());
document.querySelectorAll("[data-close-dialog]").forEach(button => {
    button.addEventListener("click", () => {
        const dialog = document.getElementById(button.dataset.closeDialog);
        if (dialog !== sessionDialog || !sessionInitializing) dialog.close();
    });
});
document.querySelectorAll(".app-dialog").forEach(dialog => {
    dialog.addEventListener("click", event => {
        if (event.target === dialog && (dialog !== sessionDialog || !sessionInitializing)) dialog.close();
    });
});

let showAir = false;

let currentBoard = null;
let currentClusterAir = null;
let currentLegal = null;
let currentCaptures = [];
let currentNumEat = null;
let currentRecommendations = [];
let currentColor = 1;
let sgfLoaded = false;
let sgfStopped = false;
let sgfPredictionRequestId = 0;
let sgfCurveRequestId = 0;
let sgfWinrateCurve = [];
let isPlayingSgf = false;
let currentSgfIndex = 0;
let sgfDone = false;
const sgfFileInput = document.getElementById("sgf-file");
const loadSgfButton = document.getElementById("load-sgf");
const stepSgfButton = document.getElementById("step-sgf");
const playSgfAllButton = document.getElementById("play-sgf-all");
const sgfProgress = document.getElementById("sgf-progress");
const sgfResultText = document.getElementById("sgf-result");
const sgfDiagnostic = document.getElementById("sgf-diagnostic");
const winrateChartSection = document.getElementById("winrate-chart-section");
const winrateChart = document.getElementById("winrate-chart");
const winrateChartStatus = document.getElementById("winrate-chart-status");
const showLegalCheckbox = document.getElementById("show-legal");
const showNumEatCheckbox = document.getElementById("show-num-eat");
canvas.addEventListener("click", function(event) {
    const rect = canvas.getBoundingClientRect();
    const border = 2;
    const x = event.clientX - rect.left - border;
    const y = event.clientY - rect.top - border;
    let col = Math.round((x-gridWidth)/gridWidth);
    let row = Math.round((y-gridWidth)/gridWidth);
    if (col < 0 || col > 18 || row < 0 || row > 18) {
        return;
    }
    playMove(row, col);
    console.log("Clicked:", row, col);
});
toggleAirButton.addEventListener("click", function() {
    showAir = !showAir;

    toggleAirButton.textContent = showAir
        ? "隐藏气数"
        : "显示气数";

    if (currentBoard !== null && currentClusterAir !== null) {
        drawBoardState(currentBoard, currentClusterAir, currentLegal, currentCaptures, currentNumEat);
    }
});

showLegalCheckbox.addEventListener("change", function() {
    if (currentBoard !== null) {
        drawBoardState(currentBoard, currentClusterAir, currentLegal, currentCaptures, currentNumEat);
    }
});

showNumEatCheckbox.addEventListener("change", function() {
    if (currentBoard !== null) {
        drawBoardState(currentBoard, currentClusterAir, currentLegal, currentCaptures, currentNumEat);
    }
});

loadSgfButton.addEventListener("click", loadSgf);
startGameButton.addEventListener("click", startGame);
passMoveButton.addEventListener("click", passMove);
showRecommendationsButton.addEventListener("click", showAiRecommendations);
predictGameResultButton.addEventListener("click", predictGameResult);
countScoreButton.addEventListener("click", countScore);
countSgfScoreButton.addEventListener("click", () => {
    sgfDialog.close();
    countScore();
});
stepSgfButton.addEventListener("click", stepSgf);
playSgfAllButton.addEventListener("click", playSgfAll);

const rect = canvas.getBoundingClientRect();
function drawgrids() {
    ctx.beginPath();
    for (let i = 0; i < 19; i++) {
        const gridWidth = canvas.width/20;
        const p = gridWidth + i * gridWidth;
        ctx.moveTo(p, gridWidth);
        ctx.lineTo(p, canvas.height - gridWidth);
        ctx.moveTo(gridWidth, p);
        ctx.lineTo(canvas.width - gridWidth, p);
    }
    ctx.stroke();
}

function drawStarPoints() {
    const starIndices = [3, 9, 15];
    ctx.fillStyle="black";
    for(let i = 0; i < starIndices.length; i++) {
        for(let j = 0; j < starIndices.length; j++) {
            ctx.beginPath();
            ctx.arc(
                30+starIndices[i]*gridWidth,
                30+starIndices[j]*gridWidth,
                5,
                0,
                Math.PI*2
            );
            ctx.fill();
        }
    }
}

function drawStone(x, y, color) {
    ctx.beginPath();
    ctx.arc(30+x*gridWidth, 30+y*gridWidth, 13, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
}

function setPolicySelection(select, value, label) {
    if (!value) return;
    if (![...select.options].some(option => option.value === value)) {
        select.add(new Option(label || "当前选择的 epoch", value));
    }
    select.value = value;
}

function populatePolicyModels(models) {
    for (const select of [policyModelBlackSelect, policyModelWhiteSelect]) {
        const previous = select.value;
        const previousLabel = select.selectedOptions[0]?.textContent;
        const basic = document.createElement("optgroup");
        basic.label = "基础模型";
        const epochs = document.createElement("optgroup");
        epochs.label = "全部历史模型";
        for (const model of models) {
            if (typeof model.id !== "string" || typeof model.label !== "string") continue;
            (model.version ? epochs : basic).append(new Option(model.label, model.id));
        }
        select.replaceChildren(basic);
        if (epochs.children.length) select.append(epochs);
        // A refresh must not silently change either side's pending selection.
        setPolicySelection(select, previous, previousLabel);
    }
}

async function loadPolicyModels(force = false) {
    if (policyCatalogLoading || sessionInitializing) return;
    policyCatalogLoading = true;
    refreshPolicyModelsButton.disabled = true;
    policyCatalogStatus.dataset.state = "loading";
    policyCatalogStatus.textContent = "正在获取模型列表…";
    try {
        const response = await fetch("/policy/models" + (force ? "?refresh=1" : ""), { cache: "no-store" });
        const data = await response.json();
        if (!response.ok || !data.success) throw new Error(data.error || "无法获取 epoch 列表。");
        if (!Array.isArray(data.models)) throw new Error("模型列表格式有误。");
        if (!sessionInitializing) populatePolicyModels(data.models);
        const count = data.models.filter(model => model.version).length;
        policyCatalogStatus.dataset.state = "";
        policyCatalogStatus.textContent = count ? `可选 ${count} 个历史模型，已合并重复副本。` : "暂未发现历史检查点；可使用基础模型。";
    } catch (error) {
        policyCatalogStatus.dataset.state = "error";
        policyCatalogStatus.textContent = error.message;
    } finally {
        policyCatalogLoading = false;
        refreshPolicyModelsButton.disabled = sessionInitializing;
    }
}

refreshPolicyModelsButton.addEventListener("click", () => loadPolicyModels(true));

function showPolicyModelState(data) {
    const labels = { supervised: "监督学习模型", rl: "强化学习模型（最新）" };
    const blackModel = data.policy_model_black ?? data.policy_model;
    const whiteModel = data.policy_model_white ?? data.policy_model;
    const blackLabel = data.policy_model_label_black || labels[blackModel];
    const whiteLabel = data.policy_model_label_white || labels[whiteModel];
    if (blackLabel && whiteLabel) {
        policyModelStatus.textContent = `黑棋模型：${blackLabel}；白棋模型：${whiteLabel}`;
    }
    // Preserve both pending choices until the user starts the new game.
    if (!sessionDialog.open) {
        setPolicySelection(policyModelBlackSelect, blackModel, blackLabel);
        setPolicySelection(policyModelWhiteSelect, whiteModel, whiteLabel);
    }
}

function showSessionState(data) {
    showPolicyModelState(data);
    resetScore();
    if (data.game_over) {
        if (data.result === "W+R" || data.result === "B+R") {
            gameStatus.textContent = data.result === "W+R" ? "黑棋投降，白棋获胜。" : "白棋投降，黑棋获胜。";
            return;
        }
        gameStatus.textContent = data.result === "two_passes"
            ? "双方连续 Pass，对局结束。"
            : "对局结束：" + (data.result || "完成");
        return;
    }
    const colorName = data.next_player === 1 ? "黑棋" : "白棋";
    const playerType = data.next_player === 1 ? data.player_black : data.player_white;
    const controller = playerType === 0 ? "AI" : "人类";
    gameStatus.textContent = "轮到" + colorName + "（" + controller + "）";
    if (data.ai_error) gameStatus.textContent += "；AI 出错：" + data.ai_error;
}

async function startGame() {
    if (sessionInitializing) return;
    const settings = {
        player_black: Number(document.getElementById("player-black").value),
        player_white: Number(document.getElementById("player-white").value),
        policy_model_black: policyModelBlackSelect.value,
        policy_model_white: policyModelWhiteSelect.value,
        komi: Number(document.getElementById("komi").value)
    };
    const dialogControls = [...sessionDialog.querySelectorAll("select, input, button")];
    const previousDisabled = dialogControls.map(control => control.disabled);
    sessionInitializing = true;
    dialogControls.forEach(control => { control.disabled = true; });
    sessionDialog.setAttribute("aria-busy", "true");
    clearAiRecommendations();
    gameStatus.textContent = "正在开始对局…";
    sessionInitMessage.dataset.state = "loading";
    sessionInitMessage.textContent = "正在准备新对局，AI 模型首次使用时可能需要下载，请稍候…";
    let selfPlayState = null;
    try {
        const response = await fetch("/session/init", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(settings)
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
            const message = data.error || "无法开始对局。";
            gameStatus.textContent = message;
            sessionInitMessage.dataset.state = "error";
            sessionInitMessage.textContent = message;
            return;
        }
        sessionInitMessage.textContent = "";
        sessionInitMessage.dataset.state = "";
        sessionDialog.close();
        currentColor = data.next_player;
        showSessionState(data);
        drawBoardState(data.board, data.cluster_air, data.islegal, [], data.num_eat);
        const bothAi = data.player_black === 0 && data.player_white === 0;
        if (bothAi && !data.ai_error && !data.game_over) selfPlayState = data;
    } catch (error) {
        const message = "无法开始对局：" + error.message;
        gameStatus.textContent = message;
        sessionInitMessage.dataset.state = "error";
        sessionInitMessage.textContent = message;
    } finally {
        sessionInitializing = false;
        sessionDialog.removeAttribute("aria-busy");
        dialogControls.forEach((control, index) => { control.disabled = previousDisabled[index]; });
        refreshPolicyModelsButton.disabled = policyCatalogLoading;
    }
    if (selfPlayState) {
        startGameButton.disabled = true;
        try {
            await runSelfPlay(selfPlayState);
        } finally {
            startGameButton.disabled = false;
        }
    }
}

async function runSelfPlay(initialState) {
    let state = initialState;
    let moves = 0;
    const moveLimit = 1000;

    while (!state.game_over && moves < moveLimit) {
        gameStatus.textContent = "AI 对局进行中…";
        await new Promise(resolve => requestAnimationFrame(resolve));
        try {
            const response = await fetch("/ai_move", { method: "POST" });
            state = await response.json();
            if (!response.ok || !state.success) {
                gameStatus.textContent = state.error || "AI 落子失败。";
                return;
            }
        } catch (error) {
            gameStatus.textContent = "AI 对局中断：" + error.message;
            return;
        }

        clearAiRecommendations();
        currentColor = state.next_player;
        showSessionState(state);
        drawBoardState(state.board, state.cluster_air, state.islegal, [], state.num_eat);
        moves += 1;
    }

    if (!state.game_over && moves >= moveLimit) {
        gameStatus.textContent = "已达到自我对局步数上限。";
    }
}

async function playMove(row, col) {
    const response = await fetch("/play", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            row: row,
            col: col,
            color: currentColor
        })
    });

    const data = await response.json();

     if (!data.success) {
        gameStatus.textContent = data.error || "非法落子。";
        return;
     }
    clearAiRecommendations();
    currentColor = Number.isInteger(data.next_player) ? data.next_player : -currentColor;
    showSessionState(data);
    drawBoardState(data.board, data.cluster_air, data.islegal, [], data.num_eat);


}

async function passMove() {
    const response = await fetch("/pass", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ color: currentColor })
    });
    const data = await response.json();
    if (!data.success) {
        gameStatus.textContent = data.error || "Pass 失败。";
        return;
    }
    clearAiRecommendations();
    if (Number.isInteger(data.next_player)) currentColor = data.next_player;
    showSessionState(data);
    drawBoardState(data.board, data.cluster_air, data.islegal, [], data.num_eat);
}

function clearAiRecommendations() {
    currentRecommendations = [];
    recommendationsText.textContent = "";
    valuePredictionText.textContent = "";
}

function resetScore(isReplay = false) {
    scoreRequestId += 1;
    scoreTerritory = null;
    scoredBoard = null;
    scoreResult.textContent = "";
    scoringReplay = isReplay;
    countScoreButton.textContent = isReplay ? "棋谱点目" : "点目";
    updateScoreButtons();
}

function updateScoreButtons(pending = false) {
    countScoreButton.disabled = pending || (scoringReplay && isPlayingSgf);
    countSgfScoreButton.disabled = !scoringReplay || countScoreButton.disabled;
}

async function countScore() {
    if (countScoreButton.disabled) return;
    const requestId = ++scoreRequestId;
    updateScoreButtons(true);
    scoreResult.textContent = "正在点目…";
    try {
        const route = scoringReplay ? "/sgf/score" : "/score";
        const response = await fetch(route, { cache: "no-store" });
        const data = await response.json();
        if (requestId !== scoreRequestId) return;
        if (!response.ok || !data.success) {
            throw new Error(data.error || "点目失败。");
        }
        scoreTerritory = data.territory;
        scoredBoard = data.scored_board;
        drawBoardState(currentBoard, currentClusterAir, currentLegal, currentCaptures, currentNumEat);
        const format = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 8 });
        const winner = data.winner === "black" ? "黑棋" : "白棋";
        const outcome = data.winner === "draw" ? "和棋" : `${winner}胜 ${format.format(data.margin)} 目`;
        scoreResult.textContent = `点目：黑棋 ${format.format(data.black_score)} 目，` +
            `白棋 ${format.format(data.white_score)} 目（含贴目 ${format.format(data.komi)}），${outcome}。` +
            "死棋保留在棋盘上，方块颜色表示点目后的归属。";
    } catch (error) {
        if (requestId === scoreRequestId) scoreResult.textContent = error.message;
    } finally {
        if (requestId === scoreRequestId) updateScoreButtons();
    }
}

async function predictGameResult() {
    predictGameResultButton.disabled = true;
    valuePredictionText.textContent = "正在预测当前局面的胜负…";
    try {
        const predictionRoute = sgfLoaded ? "/sgf/prediction" : "/value/prediction";
        const response = await fetch(predictionRoute);
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.error || "无法预测当前胜负。");
        }
        const black = (data.black_win_probability * 100).toFixed(1);
        const white = (data.white_win_probability * 100).toFixed(1);
        const winner = data.predicted_winner === "black" ? "黑棋" : "白棋";
        valuePredictionText.textContent =
            `模型预测 ${winner} 更有可能获胜；黑胜概率 ${black}%，白胜概率 ${white}%。`;
    } catch (error) {
        valuePredictionText.textContent = error.message;
    } finally {
        predictGameResultButton.disabled = false;
    }
}

function goPointName(row, col) {
    const columns = "ABCDEFGHJKLMNOPQRST";
    return `${columns[col]}${19 - row}（行 ${row}，列 ${col}）`;
}

async function showAiRecommendations() {
    showRecommendationsButton.disabled = true;
    recommendationsText.textContent = "正在计算 AI 推荐位置…";
    try {
        const response = await fetch("/ai/recommendations");
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.error || "无法获取 AI 推荐。");
        }
        if (scoringReplay) resetScore();
        currentRecommendations = data.recommendations || [];
        drawBoardState(data.board, data.cluster_air, data.islegal, [], data.num_eat);
        if (currentRecommendations.length === 0) {
            recommendationsText.textContent = "当前没有可推荐的合法落点。";
        } else {
            const side = data.color === 1 ? "黑棋" : "白棋";
            recommendationsText.textContent = `${side} AI 推荐：` + currentRecommendations
                .map((point, index) => `${index + 1}. ${goPointName(point.row, point.col)}（分数 ${point.score.toFixed(3)}）`)
                .join("；");
        }
    } catch (error) {
        currentRecommendations = [];
        recommendationsText.textContent = error.message;
        if (currentBoard !== null) {
            drawBoardState(currentBoard, currentClusterAir, currentLegal, currentCaptures, currentNumEat);
        }
    } finally {
        showRecommendationsButton.disabled = false;
    }
}

function drawBoardState(board, clusterAir, legal = null, captures = [], numEat = null) {
    currentBoard = board;
    currentClusterAir = clusterAir;
    currentLegal = legal;
    currentCaptures = captures;
    currentNumEat = numEat;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (showAir) {
        drawAir(board, clusterAir);
    }

    drawgrids();

    if (showLegalCheckbox.checked && legal !== null) {
        drawLegalPoints(legal);
    }
    
    drawStarPoints();

    for (let row = 0; row < 19; row++) {
        for (let col = 0; col < 19; col++) {
            if (board[row][col] === 1) {
                drawStone(col, row, "black");
            } else if (board[row][col] === -1) {
                drawStone(col, row, "white");
            }
        }
    }

    drawRecommendationMarkers();

    drawCaptureHighlights(captures);
    if (showNumEatCheckbox.checked && numEat !== null) {
        drawNumEat(numEat);
    }
    drawTerritoryMarkers();
}

function drawTerritoryMarkers() {
    if (!scoreTerritory || !scoredBoard) return;
    const size = gridWidth * 0.3;
    ctx.save();
    ctx.strokeStyle = "#33291c";
    ctx.lineWidth = 1;
    for (let row = 0; row < 19; row++) {
        for (let col = 0; col < 19; col++) {
            const owner = scoreTerritory[row][col];
            if (scoredBoard[row][col] !== 0 || owner === 0) continue;
            const x = (col + 1) * gridWidth - size / 2;
            const y = (row + 1) * gridWidth - size / 2;
            ctx.fillStyle = owner === 1 ? "#181818" : "#fffdf8";
            ctx.fillRect(x, y, size, size);
            ctx.strokeRect(x, y, size, size);
        }
    }
    ctx.restore();
}

function drawRecommendationMarkers() {
    currentRecommendations.forEach((point, index) => {
        const x = 30 + point.col * gridWidth;
        const y = 30 + point.row * gridWidth;
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255, 255, 255, 0.92)";
        ctx.fill();
        ctx.strokeStyle = "#b33a20";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "#8d2415";
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(index + 1), x, y);
    });
}

function drawNumEat(numEat) {
    // num_eat 的形状是 [黑方层/白方层][row][col]。
    for (let row = 0; row < 19; row++) {
        for (let col = 0; col < 19; col++) {
            const blackCount = Number(numEat[0][row][col]);
            const whiteCount = Number(numEat[1][row][col]);
            if (blackCount <= 0 && whiteCount <= 0) continue;

            const x = 30 + col * gridWidth;
            const y = 30 + row * gridWidth;
            const labels = [];
            if (blackCount > 0) labels.push({ text: `B${blackCount}`, color: "#111" });
            if (whiteCount > 0) labels.push({ text: `W${whiteCount}`, color: "#c22" });

            ctx.font = "bold 10px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            labels.forEach((label, index) => {
                const labelY = y + (labels.length === 1 ? 0 : (index === 0 ? -6 : 6));
                ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
                ctx.fillRect(x - 11, labelY - 6, 22, 12);
                ctx.fillStyle = label.color;
                ctx.fillText(label.text, x, labelY);
            });
        }
    }
}

function drawCaptureHighlights(captures) {
    for (const [row, col] of captures) {
        const x = 30 + col * gridWidth;
        const y = 30 + row * gridWidth;
        ctx.beginPath();
        ctx.arc(x, y, 9, 0, Math.PI * 2);
        ctx.strokeStyle = "#e04716";
        ctx.lineWidth = 3;
        ctx.stroke();
    }
}

function drawLegalPoints(legal) {
    ctx.fillStyle = "rgba(20, 150, 60, 0.65)";
    for (let row = 0; row < 19; row++) {
        for (let col = 0; col < 19; col++) {
            if (legal[row][col] === 1 && currentBoard[row][col] === 0) {
                ctx.beginPath();
                ctx.arc(30 + col * gridWidth, 30 + row * gridWidth, 3, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }
}

function showSgfState(data) {
    resetScore(true);
    clearAiRecommendations();
    currentSgfIndex = data.index || 0;
    sgfDone = Boolean(data.done);
    sgfResultText.textContent = describeSgfResult(data.result);
    if (sgfWinrateCurve.length > currentSgfIndex) showCurvePrediction();
    const captures = data.move?.captured || [];
    drawBoardState(data.board, data.cluster_air, data.islegal, captures, data.num_eat);
    sgfProgress.textContent = data.message || `已处理 ${data.index}/${data.total} 手`;
    const move = data.move;
    if (move && move.diagnostic) {
        const d = move.diagnostic;
        sgfStopped = true;
        sgfDiagnostic.textContent = [
            `非法落子：${d.color}，SGF 节点 ${d.sgf_node}，第 ${d.move_number} 手`,
            `SGF 坐标：${d.raw}；数组坐标：row=${d.row}, col=${d.col}`,
            `落子前目标点棋子值：${d.board_value_before}；islegal：${d.islegal_before}`,
            `count, black_eat, white_eat: ${d.count}；${d.black_eat}；${d.white_eat}`,
            `相邻棋子（黑=1，白=-1，空=0）：${JSON.stringify(d.neighbors)}`,
            `相邻 cluster_air：${JSON.stringify(d.neighbor_cluster_air)}`,
            `当前合法点图显示的是 Game.islegal。失败后棋盘停留在落子前状态。`,
        ].join("\n");
    } else if (move) {
        const capturedText = move.captured?.length
            ? move.captured.map(([row, col]) => `(${row}, ${col})`).join("、")
            : "无";
        sgfDiagnostic.textContent = [
            `最近一步：节点 ${move.node}，${move.color_name}，${move.pass ? "pass" : `(${move.row}, ${move.col})`}，结果：${move.success ? "成功" : "失败"}`,
            `本手实际提子：${move.captured?.length || 0} 子；坐标（row, col）：${capturedText}`,
            "棋盘上橙红色圆圈标出本手提掉的位置。",
        ].join("\n");
    }
    stepSgfButton.disabled = !sgfLoaded || sgfStopped || sgfDone || isPlayingSgf;
    playSgfAllButton.disabled = !sgfLoaded || sgfStopped || sgfDone || isPlayingSgf;
    drawWinrateChart();
}

function describeSgfResult(result) {
    if (!result) return "棋谱结果：未记录";
    const normalized = String(result).toUpperCase();
    if (normalized.startsWith("B+")) return `棋谱结果：黑棋胜（${result}）`;
    if (normalized.startsWith("W+")) return `棋谱结果：白棋胜（${result}）`;
    if (normalized === "0" || normalized === "DRAW") return `棋谱结果：和棋（${result}）`;
    return `棋谱结果：${result}`;
}

function showCurvePrediction() {
    const point = sgfWinrateCurve[currentSgfIndex];
    if (!point) return;
    const black = (point.black_win_probability * 100).toFixed(1);
    const white = (100 - Number(black)).toFixed(1);
    const winner = point.black_win_probability >= 0.5 ? "黑棋" : "白棋";
    valuePredictionText.textContent =
        `当前局面预测：${winner}更有可能获胜；黑胜 ${black}%，白胜 ${white}%。`;
}

async function loadWinrateCurve() {
    const requestId = ++sgfCurveRequestId;
    winrateChartSection.hidden = false;
    winrateChartStatus.textContent = "正在计算整局胜率变化…";
    sgfWinrateCurve = [];
    try {
        const response = await fetch("/sgf/winrate-curve", {
            signal: AbortSignal.timeout(180000),
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.error || "无法计算胜率曲线。");
        }
        if (requestId !== sgfCurveRequestId) return;
        sgfWinrateCurve = data.curve || [];
        drawWinrateChart();
        showCurvePrediction();
        winrateChartStatus.textContent = "横轴为手数，纵轴为黑棋胜率；圆点表示当前回放位置。";
    } catch (error) {
        if (requestId !== sgfCurveRequestId) return;
        winrateChartStatus.textContent = error.name === "TimeoutError"
            ? "计算超过 3 分钟。请查看 Flask 终端中的模型加载或推理日志。"
            : error.name === "AbortError"
                ? "计算已取消。"
                : error.message;
    }
}

function drawWinrateChart() {
    if (sgfWinrateCurve.length === 0) return;

    const chartContext = winrateChart.getContext("2d");
    const width = winrateChart.width;
    const height = winrateChart.height;
    const padding = { left: 54, right: 20, top: 18, bottom: 36 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const lastMove = sgfWinrateCurve[sgfWinrateCurve.length - 1].move || 1;
    const currentMove = Math.min(sgfWinrateCurve.length - 1, currentSgfIndex);
    const x = move => padding.left + (move / Math.max(1, lastMove)) * plotWidth;
    const y = probability => padding.top + (1 - probability) * plotHeight;

    chartContext.clearRect(0, 0, width, height);
    chartContext.font = "12px Arial";
    chartContext.fillStyle = "#463522";
    chartContext.strokeStyle = "#d8c8aa";
    chartContext.lineWidth = 1;

    [0, 0.25, 0.5, 0.75, 1].forEach(probability => {
        const lineY = y(probability);
        chartContext.beginPath();
        chartContext.moveTo(padding.left, lineY);
        chartContext.lineTo(width - padding.right, lineY);
        chartContext.stroke();
        chartContext.textAlign = "right";
        chartContext.textBaseline = "middle";
        chartContext.fillText(`${Math.round(probability * 100)}%`, padding.left - 7, lineY);
    });

    chartContext.strokeStyle = "#6b4423";
    chartContext.beginPath();
    chartContext.moveTo(padding.left, padding.top);
    chartContext.lineTo(padding.left, height - padding.bottom);
    chartContext.lineTo(width - padding.right, height - padding.bottom);
    chartContext.stroke();

    chartContext.strokeStyle = "#1b5e20";
    chartContext.lineWidth = 2;
    chartContext.beginPath();
    sgfWinrateCurve.forEach((point, index) => {
        const pointX = x(point.move);
        const pointY = y(point.black_win_probability);
        if (index === 0) chartContext.moveTo(pointX, pointY);
        else chartContext.lineTo(pointX, pointY);
    });
    chartContext.stroke();

    const selected = sgfWinrateCurve[currentMove];
    if (selected) {
        const pointX = x(selected.move);
        const pointY = y(selected.black_win_probability);
        chartContext.fillStyle = "#b33a20";
        chartContext.beginPath();
        chartContext.arc(pointX, pointY, 5, 0, Math.PI * 2);
        chartContext.fill();
        chartContext.textAlign = "left";
        chartContext.textBaseline = "bottom";
        chartContext.fillText(
            `第 ${selected.move} 手：黑胜 ${(selected.black_win_probability * 100).toFixed(1)}%`,
            Math.min(pointX + 8, width - 175),
            Math.max(pointY - 8, 14),
        );
    }

    chartContext.textAlign = "center";
    chartContext.textBaseline = "top";
    chartContext.fillStyle = "#463522";
    chartContext.fillText("手数", padding.left + plotWidth / 2, height - padding.bottom + 13);
}


async function loadSgf() {
    const file = sgfFileInput.files[0];
    if (!file) {
        sgfProgress.textContent = "请先选择一个 .sgf 文件。";
        return;
    }
    const body = new FormData();
    body.append("sgf", file);
    loadSgfButton.disabled = true;
    sgfDiagnostic.textContent = "";
    try {
        const response = await fetch("/sgf/load", { method: "POST", body });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "载入失败");
        sgfLoaded = true;
        sgfStopped = false;
        currentSgfIndex = 0;
        sgfDone = false;
        showSgfState(data);
        stepSgfButton.disabled = false;
        playSgfAllButton.disabled = false;
        loadWinrateCurve();
    } catch (error) {
        sgfLoaded = false;
        stepSgfButton.disabled = true;
        playSgfAllButton.disabled = true;
        sgfProgress.textContent = `载入失败：${error.message}`;
    } finally {
        loadSgfButton.disabled = false;
    }
}

async function stepSgf() {
    if (!sgfLoaded || sgfStopped || isPlayingSgf) return;
    stepSgfButton.disabled = true;
    try {
        const response = await fetch("/sgf/step", { method: "POST" });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "回放失败");
        showSgfState(data);
    } catch (error) {
        sgfProgress.textContent = `回放失败：${error.message}`;
    } finally {
        stepSgfButton.disabled = !sgfLoaded || sgfStopped || sgfDone;
        playSgfAllButton.disabled = !sgfLoaded || sgfStopped || sgfDone;
    }
}

async function playSgfAll() {
    if (!sgfLoaded || sgfStopped || isPlayingSgf) return;

    isPlayingSgf = true;
    updateScoreButtons();
    playSgfAllButton.disabled = true;
    stepSgfButton.disabled = true;
    try {
        while (sgfLoaded && !sgfStopped) {
            const response = await fetch("/sgf/step", { method: "POST" });
            const data = await response.json();
            if (!response.ok || !data.ok) throw new Error(data.error || "回放失败");
            showSgfState(data);
            if (data.done || sgfStopped) break;
            await new Promise(resolve => setTimeout(resolve, 70));
        }
    } catch (error) {
        sgfProgress.textContent = `回放失败：${error.message}`;
    } finally {
        isPlayingSgf = false;
        updateScoreButtons();
        stepSgfButton.disabled = !sgfLoaded || sgfStopped || sgfDone;
        playSgfAllButton.disabled = !sgfLoaded || sgfStopped || sgfDone;
    }
}
function drawAir(board, clusterAir) {
    for (let row = 0; row < 19; row++) {
        for (let col = 0; col < 19; col++) {

            // 空位置不显示气数
            if (board[row][col] === 0) {
                continue;
            }

            const airCount = clusterAir[row][col];

            // 交叉点的中心坐标
            const x = 30 + col * gridWidth;
            const y = 30 + row * gridWidth;

            // 根据气数选择半透明颜色
            if (airCount === 0) {
                ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
            } else if (airCount === 1) {
                ctx.fillStyle = "rgba(255, 0, 0, 0.4)";
            } else if (airCount === 2) {
                ctx.fillStyle = "rgba(255, 140, 0, 0.4)";
            } else if (airCount === 3) {
                ctx.fillStyle = "rgba(255, 220, 0, 0.4)";
            } else {
                ctx.fillStyle = "rgba(0, 180, 0, 0.4)";
            }

            // 在交叉点上绘制半透明方格
            ctx.fillRect(
                x - gridWidth / 2,
                y - gridWidth / 2,
                gridWidth,
                gridWidth
            );
        }
    }
}

async function loadBoard() {
    const response = await fetch("/state");
    const data = await response.json();
    if (Number.isInteger(data.next_player)) currentColor = data.next_player;

    if (gameStatus) showSessionState(data);
    drawBoardState(data.board, data.cluster_air, data.islegal, [], data.num_eat);
}

loadBoard();
