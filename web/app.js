(() => {
  "use strict";

  const PITS = 6;
  const STONES = 4;
  const STORE_0 = PITS;
  const STORE_1 = 2 * PITS + 1;
  const HUMAN_DELAY_MS = 220;

  const elements = {};
  let state = newGame();
  let humanPlayer = 0;
  let opponent = "greedy";
  let history = [];
  let busy = false;

  function newGame() {
    return {
      board: [...Array(PITS).fill(STONES), 0, ...Array(PITS).fill(STONES), 0],
      currentPlayer: 0,
    };
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
    const mover = gameState.currentPlayer;
    const source = pitIndex(mover, action);
    let stones = board[source];
    board[source] = 0;

    let index = source;
    while (stones > 0) {
      index = (index + 1) % board.length;
      if (index === storeIndex(1 - mover)) continue;
      board[index] += 1;
      stones -= 1;
    }

    const ownStore = storeIndex(mover);
    let capturedStones = false;
    if (pitIndices(mover).includes(index) && board[index] === 1) {
      const opposite = oppositeIndex(index);
      const captured = board[opposite];
      if (captured > 0) {
        board[opposite] = 0;
        board[index] = 0;
        board[ownStore] += captured + 1;
        capturedStones = true;
      }
    }

    const nextPlayer = index === ownStore || capturedStones ? mover : 1 - mover;
    if (sideEmpty(board, 0) || sideEmpty(board, 1)) {
      sweepRemaining(board);
    }

    return {
      board,
      currentPlayer: nextPlayer,
      lastMove: {
        player: mover,
        action,
        extraTurn: nextPlayer === mover && !sideEmpty(board, 0) && !sideEmpty(board, 1),
        captured: capturedStones,
      },
    };
  }

  function sweepRemaining(board) {
    for (const player of [0, 1]) {
      const store = storeIndex(player);
      for (const index of pitIndices(player)) {
        board[store] += board[index];
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
    elements.humanPlayer = document.querySelector("#human-player");
    elements.newGame = document.querySelector("#new-game");

    elements.opponent.addEventListener("change", () => {
      opponent = elements.opponent.value;
      maybeAgentMove();
    });
    elements.humanPlayer.addEventListener("change", () => {
      humanPlayer = Number(elements.humanPlayer.value);
      resetGame();
    });
    elements.newGame.addEventListener("click", resetGame);

    render();
    maybeAgentMove();
  }

  function resetGame() {
    state = newGame();
    history = [];
    busy = false;
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
    window.setTimeout(() => {
      const action = opponent === "random" ? chooseRandom(state) : chooseGreedy(state);
      playAction(action);
      busy = false;
      render();
      maybeAgentMove();
    }, HUMAN_DELAY_MS);
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

    renderStore(elements.northStore, state.board[STORE_1], "North store");
    renderStore(elements.southStore, state.board[STORE_0], "South store");
    renderPitRows();
    renderHistory();
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
    button.style.gridColumn = String(player === 0 ? action + 2 : PITS - action + 1);
    button.style.gridRow = String(player === 0 ? 2 : 1);
    button.disabled = !legal;
    button.setAttribute("aria-label", `${playerName(player)} pit ${action + 1}, ${state.board[index]} stones`);
    button.addEventListener("click", () => onPitClick(action));
    button.appendChild(label(`Pit ${action + 1}`));
    button.appendChild(stonesLayer(state.board[index], `pit-${player}-${action}`));
    button.appendChild(count(state.board[index]));
    return button;
  }

  function renderStore(container, stones, name) {
    container.innerHTML = "";
    container.setAttribute("aria-label", `${name}, ${stones} stones`);
    container.appendChild(label(name.replace(" store", "")));
    container.appendChild(stonesLayer(stones, name));
    container.appendChild(count(stones));
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

  function stonesLayer(amount, salt) {
    const layer = document.createElement("span");
    layer.className = "stones";
    const visible = Math.min(amount, 18);
    for (let i = 0; i < visible; i += 1) {
      const stone = document.createElement("span");
      stone.className = "stone";
      const point = stonePoint(i, amount, salt);
      stone.style.left = `${point.x}%`;
      stone.style.top = `${point.y}%`;
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
    return {
      x: 8 + (hash % 76),
      y: 8 + ((hash >>> 8) % 76),
    };
  }

  function statusText() {
    if (isTerminal(state)) {
      if (score(0) === score(1)) return `Draw, ${score(0)}-${score(1)}`;
      return `${score(0) > score(1) ? "South" : "North"} wins, ${score(0)}-${score(1)}`;
    }
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

