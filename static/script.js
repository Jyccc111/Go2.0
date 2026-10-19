const canvas = document.getElementById("go-board");
const ctx = canvas.getContext("2d");
const gridWidth = canvas.width/20;
const toggleAirButton = document.getElementById("toggle-air");

let showAir = false;

let currentBoard = null;
let currentClusterAir = null;
let currentLegal = null;
let currentCaptures = [];
let currentNumEat = null;
let currentColor = 1;
let sgfLoaded = false;
let sgfStopped = false;
const sgfFileInput = document.getElementById("sgf-file");
const loadSgfButton = document.getElementById("load-sgf");
const stepSgfButton = document.getElementById("step-sgf");
const sgfProgress = document.getElementById("sgf-progress");
const sgfDiagnostic = document.getElementById("sgf-diagnostic");
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
stepSgfButton.addEventListener("click", stepSgf);

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
        console.log("Illegal move!");
        return;
     }
    currentColor = -currentColor;
    drawBoardState(data.board, data.cluster_air, data.islegal, [], data.num_eat);


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

    drawCaptureHighlights(captures);
    if (showNumEatCheckbox.checked && numEat !== null) {
        drawNumEat(numEat);
    }
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
    stepSgfButton.disabled = !sgfLoaded || sgfStopped || data.done;
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
        showSgfState(data);
        stepSgfButton.disabled = false;
    } catch (error) {
        sgfLoaded = false;
        stepSgfButton.disabled = true;
        sgfProgress.textContent = `载入失败：${error.message}`;
    } finally {
        loadSgfButton.disabled = false;
    }
}

async function stepSgf() {
    if (!sgfLoaded || sgfStopped) return;
    stepSgfButton.disabled = true;
    try {
        const response = await fetch("/sgf/step", { method: "POST" });
        const data = await response.json();
        if (!response.ok || !data.ok) throw new Error(data.error || "回放失败");
        showSgfState(data);
    } catch (error) {
        sgfProgress.textContent = `回放失败：${error.message}`;
    } finally {
        if (sgfLoaded && !sgfStopped) stepSgfButton.disabled = false;
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

    drawBoardState(data.board, data.cluster_air, data.islegal, [], data.num_eat);
}

loadBoard();
