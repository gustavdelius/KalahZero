(() => {
  "use strict";

  const PITS = 6;
  const STONES = 4;
  const STONE_COLORS = ['#d4a017', '#c85a10', '#a82010', '#2e5e2e', '#8a7860', '#c8a050'];
  const STORE_0 = PITS;
  const STORE_1 = 2 * PITS + 1;
  const HUMAN_DELAY_MS = 220;

  const elements = {};
  let state = newGame();
  let humanPlayer = 0;
  let opponent = "greedy";
  let history = [];
  let busy = false;
  let agentError = "";

  function randomColorIdx() {
    return Math.floor(Math.random() * STONE_COLORS.length);
  }

  function newGame() {
    const board = [...Array(PITS).fill(STONES), 0, ...Array(PITS).fill(STONES), 0];
    const colors = board.map(count => Array.from({ length: count }, randomColorIdx));
    return { board, colors, currentPlayer: 0 };
  }

  function storeIndex(player) {
    return player === 0 ? STORE_0 : STORE_1;
  }

  function pitIndices(player) {
    if (player === 0) return [0, 1, 2, 3, 4, 5];
    return [7, 8, 9, 10, 11, 12];
  }

  function pitIndex(player, action) {
    return player === 0 ? action : PITS + 1 + action;
  }

  function actionForIndex(player, index) {
    return player === 0 ? index : index - (PITS + 1);
  }

  function oppositeIndex(index) {
    return 2 * PITS - index;
  }

  function sideEmpty(board, player) {
    return pitIndices(player).every((index) => board[index] === 0);
  }

  function isTerminal(gameState) {
    return sideEmpty(gameState.board, 0) || sideEmpty(gameState.board, 1);
  }

  function legalActions(gameState) {
    if (isTerminal(gameState)) return [];
    return pitIndices(gameState.currentPlayer)
      .filter((index) => gameState.board[index] > 0)
      .map((index) => actionForIndex(gameState.currentPlayer, index));
  }

  function applyMove(gameState, action) {
    if (!legalActions(gameState).includes(action)) {
      throw new Error(`Illegal action ${action}`);
    }

    const board = [...gameState.board];
    const colors = gameState.colors.map(arr => [...arr]);
    const mover = gameState.currentPlayer;
    const source = pitIndex(mover, action);
    let stones = board[source];
    const pool = colors[source];  // colors of the stones being lifted
    board[source] = 0;
    colors[source] = [];

    let index = source;
    let poolIdx = 0;
    while (stones > 0) {
      index = (index + 1) % board.length;
      if (index === storeIndex(1 - mover)) continue;
      board[index] += 1;
      colors[index].push(pool[poolIdx++]);
      stones -= 1;
    }

    const ownStore = storeIndex(mover);
    let capturedStones = false;
    if (pitIndices(mover).includes(index) && board[index] === 1) {
      const opposite = oppositeIndex(index);
      const captured = board[opposite];
      if (captured > 0) {
        colors[ownStore].push(...colors[index], ...colors[opposite]);
        colors[index] = [];
        colors[opposite] = [];
        board[opposite] = 0;
        board[index] = 0;
        board[ownStore] += captured + 1;
        capturedStones = true;
      }
    }

    const nextPlayer = index === ownStore || capturedStones ? mover : 1 - mover;
    if (sideEmpty(board, 0) || sideEmpty(board, 1)) {
      sweepRemaining(board, colors);
    }

    return {
      board,
      colors,
      currentPlayer: nextPlayer,
      lastMove: {
        player: mover,
        action,
        extraTurn: nextPlayer === mover && !sideEmpty(board, 0) && !sideEmpty(board, 1),
        captured: capturedStones,
      },
    };
  }

  function sweepRemaining(board, colors) {
    for (const player of [0, 1]) {
      const store = storeIndex(player);
      for (const index of pitIndices(player)) {
        board[store] += board[index];
        colors[store].push(...colors[index]);
        colors[index] = [];
        board[index] = 0;
      }
    }
  }

  function score(player) {
    return state.board[storeIndex(player)];
  }

  function chooseRandom(gameState) {
    const legal = legalActions(gameState);
    return legal[Math.floor(Math.random() * legal.length)];
  }

  function chooseGreedy(gameState) {
    const player = gameState.currentPlayer;
    return legalActions(gameState).reduce((best, action) => {
      const bestState = applyMove(gameState, best);
      const nextState = applyMove(gameState, action);
      return margin(nextState, player) > margin(bestState, player) ? action : best;
    });
  }

  function margin(gameState, player) {
    return gameState.board[storeIndex(player)] - gameState.board[storeIndex(1 - player)];
  }

  function isHumanTurn() {
    return !isTerminal(state) && state.currentPlayer === humanPlayer;
  }

  function playerName(player) {
    return player === 0 ? "South" : "North";
  }

  function init() {
    elements.board = document.querySelector("#board");
    elements.northRow = document.querySelector("#north-row");
    elements.southRow = document.querySelector("#south-row");
    elements.northStore = document.querySelector("#store-north");
    elements.southStore = document.querySelector("#store-south");
    elements.scoreNorth = document.querySelector("#score-north");
    elements.scoreSouth = document.querySelector("#score-south");
    elements.status = document.querySelector("#status");
    elements.history = document.querySelector("#history");
    elements.positionSummary = document.querySelector("#position-summary");
    elements.opponent = document.querySelector("#opponent");
    elements.agentUrlControl = document.querySelector("#agent-url-control");
    elements.agentUrl = document.querySelector("#agent-url");
    elements.humanPlayer = document.querySelector("#human-player");
    elements.newGame = document.querySelector("#new-game");

    elements.opponent.addEventListener("change", () => {
      opponent = elements.opponent.value;
      agentError = "";
      updateAgentUrlControl();
      maybeAgentMove();
    });
    elements.humanPlayer.addEventListener("change", () => {
      humanPlayer = Number(elements.humanPlayer.value);
      resetGame();
    });
    elements.newGame.addEventListener("click", resetGame);

    render();
    updateAgentUrlControl();
    maybeAgentMove();
  }

  function resetGame() {
    state = newGame();
    history = [];
    busy = false;
    agentError = "";
    render();
    maybeAgentMove();
  }

  function onPitClick(action) {
    if (busy || !isHumanTurn() || !legalActions(state).includes(action)) return;
    playAction(action);
    maybeAgentMove();
  }

  function maybeAgentMove() {
    if (busy || isTerminal(state) || isHumanTurn()) return;
    busy = true;
    render();
    window.setTimeout(async () => {
      try {
        const action = await chooseOpponentAction(state);
        agentError = "";
        playAction(action);
        busy = false;
        render();
        maybeAgentMove();
      } catch (error) {
        agentError = error.message || String(error);
        busy = false;
        render();
      }
    }, HUMAN_DELAY_MS);
  }

  async function chooseOpponentAction(gameState) {
    if (opponent === "random") return chooseRandom(gameState);
    if (opponent === "greedy") return chooseGreedy(gameState);
    if (opponent === "trained") return requestTrainedAgentMove(gameState);
    throw new Error(`Unknown opponent ${opponent}`);
  }

  async function requestTrainedAgentMove(gameState) {
    const endpoint = elements.agentUrl.value.trim();
    if (!endpoint) throw new Error("Agent server URL is empty");
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        board: gameState.board,
        current_player: gameState.currentPlayer,
        pits: PITS,
      }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || `Agent server returned ${response.status}`);
    }
    if (!legalActions(gameState).includes(payload.action)) {
      throw new Error(`Agent server returned illegal action ${payload.action}`);
    }
    return payload.action;
  }

  function playAction(action) {
    const before = state.currentPlayer;
    state = applyMove(state, action);
    const tags = [];
    if (state.lastMove.captured) tags.push("capture");
    if (state.lastMove.extraTurn) tags.push("again");
    history.unshift(`${playerName(before)} ${action + 1}${tags.length ? ` (${tags.join(", ")})` : ""}`);
    history = history.slice(0, 18);
    render();
  }

  function render() {
    elements.scoreNorth.textContent = String(score(1));
    elements.scoreSouth.textContent = String(score(0));
    elements.status.textContent = statusText();
    elements.positionSummary.textContent = summaryText();

    renderStore(elements.northStore, state.board[STORE_1], "North store", state.colors[STORE_1]);
    renderStore(elements.southStore, state.board[STORE_0], "South store", state.colors[STORE_0]);
    renderPitRows();
    renderHistory();
  }

  function updateAgentUrlControl() {
    elements.agentUrlControl.classList.toggle("hidden", opponent !== "trained");
  }

  function renderPitRows() {
    elements.northRow.innerHTML = "";
    elements.southRow.innerHTML = "";

    for (let action = PITS - 1; action >= 0; action -= 1) {
      elements.northRow.appendChild(createPit(1, action));
    }
    for (let action = 0; action < PITS; action += 1) {
      elements.southRow.appendChild(createPit(0, action));
    }
  }

  function createPit(player, action) {
    const index = pitIndex(player, action);
    const button = document.createElement("button");
    const legal = isHumanTurn() && player === humanPlayer && legalActions(state).includes(action);
    button.type = "button";
    button.className = `pit-button ${legal ? "legal" : "disabled"}`;
    // Vertical board: south (player 0) in left column, pit 0 at bottom (row 7); north in right column, pit 0 at top (row 2)
    button.style.gridColumn = String(player === 0 ? 1 : 2);
    button.style.gridRow = String(player === 0 ? PITS + 1 - action : action + 2);
    button.disabled = !legal;
    button.setAttribute("aria-label", `${playerName(player)} pit ${action + 1}, ${state.board[index]} stones`);
    button.addEventListener("click", () => onPitClick(action));
    button.appendChild(stonesLayer(state.board[index], `pit-${player}-${action}`, state.colors[index]));
    return button;
  }

  function renderStore(container, stones, name, stoneColors) {
    container.innerHTML = "";
    container.setAttribute("aria-label", `${name}, ${stones} stones`);
    container.appendChild(stonesLayer(stones, name, stoneColors));
  }

  function label(text) {
    const span = document.createElement("span");
    span.className = "pit-label";
    span.textContent = text;
    return span;
  }

  function count(value) {
    const span = document.createElement("span");
    span.className = "count";
    span.textContent = String(value);
    return span;
  }

  function stonesLayer(amount, salt, stoneColors) {
    const layer = document.createElement("span");
    layer.className = "stones";
    const visible = Math.min(amount, 18);
    for (let i = 0; i < visible; i += 1) {
      const stone = document.createElement("span");
      stone.className = "stone";
      const { x, y } = stonePoint(i, amount, salt);
      stone.style.left = `${x}%`;
      stone.style.top = `${y}%`;
      stone.style.setProperty("--stone-color", STONE_COLORS[stoneColors[i] % STONE_COLORS.length]);
      layer.appendChild(stone);
    }
    return layer;
  }

  function stonePoint(index, amount, salt) {
    let hash = 0;
    const text = `${salt}-${index}-${amount}`;
    for (let i = 0; i < text.length; i += 1) {
      hash = (hash * 31 + text.charCodeAt(i)) >>> 0;
    }
    return { x: 8 + (hash % 76), y: 8 + ((hash >>> 8) % 76) };
  }

  function statusText() {
    if (isTerminal(state)) {
      if (score(0) === score(1)) return `Draw, ${score(0)}-${score(1)}`;
      return `${score(0) > score(1) ? "South" : "North"} wins, ${score(0)}-${score(1)}`;
    }
    if (agentError) return `Agent server error: ${agentError}`;
    if (busy) return `${playerName(state.currentPlayer)} thinking`;
    return `${playerName(state.currentPlayer)} to move`;
  }

  function summaryText() {
    const south = state.board.slice(0, PITS).join(" ");
    const north = state.board.slice(PITS + 1, STORE_1).join(" ");
    return `South [${south}] | North [${north}]`;
  }

  function renderHistory() {
    elements.history.innerHTML = "";
    for (const item of history) {
      const li = document.createElement("li");
      li.textContent = item;
      elements.history.appendChild(li);
    }
  }

  if (typeof window !== "undefined") {
    window.KalahWeb = { newGame, applyMove, legalActions, isTerminal };
    window.addEventListener("DOMContentLoaded", init);
  }

  if (typeof module !== "undefined") {
    module.exports = { newGame, applyMove, legalActions, isTerminal, PITS, STORE_0, STORE_1 };
  }
})();
