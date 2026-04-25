(() => {
  "use strict";

  const PITS = 6;
  const DEFAULT_STONES = 6;
  const STONE_COLORS = ['#d4a017', '#c85a10', '#a82010', '#2e5e2e', '#8a7860', '#c8a050'];
  const STORE_0 = PITS;
  const STORE_1 = 2 * PITS + 1;
  const HUMAN_DELAY_MS = 220;
  // Indexed by slider value 1–5 (slow → fast).
  const SPEED_DELAYS = [800, 500, 300, 150, 80];
  const SCRIPT_BASE_URL = (() => {
    if (typeof document !== "undefined" && document.currentScript) {
      return new URL(".", document.currentScript.src);
    }
    if (typeof window !== "undefined" && window.location) {
      return new URL(".", window.location.href);
    }
    return new URL("file:///");
  })();

  const elements = {};
  let startingStones = DEFAULT_STONES;
  let state = newGame();
  let humanPlayer = 0;
  let opponent = "greedy";
  let history = [];
  let busy = false;
  let animating = false;
  let agentError = "";
  let agentWorker = null;
  let agentRequestId = 0;
  const agentRequests = new Map();

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function stoneDropDelay() {
    const val = elements.animSpeed ? Number(elements.animSpeed.value) : 3;
    return SPEED_DELAYS[Math.min(Math.max(val - 1, 0), SPEED_DELAYS.length - 1)];
  }

  function randomColorIdx() {
    return Math.floor(Math.random() * STONE_COLORS.length);
  }

  function newGame(stones = startingStones) {
    const board = [...Array(PITS).fill(stones), 0, ...Array(PITS).fill(stones), 0];
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

  // Returns an array of intermediate board snapshots during sowing,
  // one per stone placed (plus step 0 for the pickup).
  function computeSowingSteps(gameState, action) {
    const steps = [];
    const board = [...gameState.board];
    const colors = gameState.colors.map(arr => [...arr]);
    const mover = gameState.currentPlayer;
    const source = pitIndex(mover, action);

    let remaining = board[source];
    const pool = colors[source].slice();

    const snap = () => ({ board: [...board], colors: colors.map(a => [...a]), activePit: source, sourcePit: source });

    // Step 0: source pit highlighted with all stones still present.
    steps.push(snap());

    let index = source;
    let poolIdx = 0;
    while (remaining > 0) {
      index = (index + 1) % board.length;
      if (index === storeIndex(1 - mover)) continue;

      // Highlight the destination before the stone lands; source still full.
      steps.push({ ...snap(), activePit: index });

      // Remove one stone from source, place in destination.
      board[source] -= 1;
      colors[source] = pool.slice(poolIdx + 1);
      board[index] += 1;
      colors[index].push(pool[poolIdx]);
      poolIdx++;
      remaining--;

      // Stone has landed; source shows one fewer stone.
      steps.push({ ...snap(), activePit: index });
    }

    return { steps, mover, landingIndex: index };
  }

  // Returns animation steps for capture and/or end-of-game sweep,
  // starting from the board state in lastStep (after sowing).
  function computePostSowSteps(lastStep, mover, landingIndex) {
    const steps = [];
    const board = [...lastStep.board];
    const colors = lastStep.colors.map(a => [...a]);
    const ownStore = storeIndex(mover);

    const snap = (extra) => ({ board: [...board], colors: colors.map(a => [...a]), ...extra });

    // Capture
    if (pitIndices(mover).includes(landingIndex) && board[landingIndex] === 1) {
      const opposite = oppositeIndex(landingIndex);
      if (board[opposite] > 0) {
        steps.push(snap({ activePit: landingIndex, capturePit: opposite }));

        colors[ownStore].push(...colors[landingIndex], ...colors[opposite]);
        colors[landingIndex] = [];
        colors[opposite] = [];
        board[ownStore] += board[opposite] + 1;
        board[opposite] = 0;
        board[landingIndex] = 0;

        steps.push(snap({ activePit: ownStore }));
      }
    }

    // Sweep
    if (sideEmpty(board, 0) || sideEmpty(board, 1)) {
      const sweepPits = [0, 1].flatMap(p => pitIndices(p).filter(i => board[i] > 0));
      if (sweepPits.length > 0) {
        steps.push(snap({ sweepPits }));

        for (const player of [0, 1]) {
          const store = storeIndex(player);
          for (const idx of pitIndices(player)) {
            board[store] += board[idx];
            colors[store].push(...colors[idx]);
            colors[idx] = [];
            board[idx] = 0;
          }
        }

        steps.push(snap({ allStoresActive: true }));
      }
    }

    return steps;
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
    elements.agentSimulationsControl = document.querySelector("#agent-simulations-control");
    elements.agentSimulations = document.querySelector("#agent-simulations");
    elements.humanPlayer = document.querySelector("#human-player");
    elements.startingStones = document.querySelector("#starting-stones");
    elements.newGame = document.querySelector("#new-game");
    elements.animSpeed = document.querySelector("#anim-speed");

    elements.opponent.addEventListener("change", () => {
      opponent = elements.opponent.value;
      agentError = "";
      updateAgentControls();
      maybeAgentMove();
    });
    elements.humanPlayer.addEventListener("change", () => {
      humanPlayer = Number(elements.humanPlayer.value);
      resetGame();
    });
    elements.startingStones.addEventListener("change", () => {
      startingStones = Number(elements.startingStones.value);
      resetGame();
    });
    elements.newGame.addEventListener("click", resetGame);

    render();
    updateAgentControls();
    maybeAgentMove();
  }

  function resetGame() {
    state = newGame(startingStones);
    history = [];
    busy = false;
    animating = false;
    agentError = "";
    render();
    maybeAgentMove();
  }

  async function onPitClick(action) {
    if (busy || !isHumanTurn() || !legalActions(state).includes(action)) return;
    busy = true;
    render();
    await playAction(action);
    busy = false;
    render();
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
        await playAction(action);
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
    const worker = getAgentWorker();
    const simulations = elements.agentSimulations ? Number(elements.agentSimulations.value) : 120;
    const payload = await new Promise((resolve, reject) => {
      const id = ++agentRequestId;
      const timeout = window.setTimeout(() => {
        agentRequests.delete(id);
        reject(new Error("Agent took too long to move"));
      }, 20000);
      agentRequests.set(id, { resolve, reject, timeout });
      worker.postMessage({
        type: "move",
        id,
        board: gameState.board,
        currentPlayer: gameState.currentPlayer,
        simulations,
      });
    });
    if (!legalActions(gameState).includes(payload.action)) {
      throw new Error(`Agent returned illegal action ${payload.action}`);
    }
    return payload.action;
  }

  function getAgentWorker() {
    if (typeof Worker === "undefined") {
      throw new Error("This browser does not support Web Workers");
    }
    if (agentWorker) return agentWorker;
    agentWorker = new Worker(new URL("agent_worker.js", SCRIPT_BASE_URL));
    agentWorker.addEventListener("message", (event) => {
      const payload = event.data || {};
      const pending = agentRequests.get(payload.id);
      if (!pending) return;
      window.clearTimeout(pending.timeout);
      agentRequests.delete(payload.id);
      if (payload.ok) {
        pending.resolve(payload);
      } else {
        pending.reject(new Error(payload.error || "Agent failed to move"));
      }
    });
    agentWorker.addEventListener("error", (error) => {
      for (const [id, pending] of agentRequests) {
        window.clearTimeout(pending.timeout);
        pending.reject(new Error(error.message || "Agent worker failed"));
        agentRequests.delete(id);
      }
    });
    return agentWorker;
  }

  async function playAction(action) {
    animating = true;
    const { steps: sowingSteps, mover, landingIndex } = computeSowingSteps(state, action);
    for (let i = 0; i < sowingSteps.length; i++) {
      if (i > 0) await sleep(stoneDropDelay());
      renderBoard(sowingSteps[i]);
    }

    const postSteps = computePostSowSteps(sowingSteps[sowingSteps.length - 1], mover, landingIndex);
    for (const step of postSteps) {
      await sleep(stoneDropDelay() * 2);
      renderBoard(step);
    }

    await sleep(stoneDropDelay());
    animating = false;

    const before = state.currentPlayer;
    state = applyMove(state, action);
    const tags = [];
    if (state.lastMove.captured) tags.push("capture");
    if (state.lastMove.extraTurn) tags.push("again");
    history.unshift(`${playerName(before)} ${action + 1}${tags.length ? ` (${tags.join(", ")})` : ""}`);
    history = history.slice(0, 18);
  }

  function render() {
    elements.scoreNorth.textContent = String(score(1));
    elements.scoreSouth.textContent = String(score(0));
    elements.status.textContent = statusText();
    elements.positionSummary.textContent = summaryText();
    renderStores(state);
    renderPitRows(state);
    renderHistory();
  }

  // Renders only the board pits and stores from a display state (used during animation).
  function renderBoard(displayState) {
    renderStores(displayState);
    renderPitRows(displayState);
  }

  function updateAgentControls() {
    elements.agentSimulationsControl.classList.toggle("hidden", opponent !== "trained");
  }

  function renderStores(boardState) {
    const activeNorth = boardState.activePit === STORE_1 || boardState.allStoresActive;
    const activeSouth = boardState.activePit === STORE_0 || boardState.allStoresActive;
    renderStore(elements.northStore, boardState.board[STORE_1], "North store", boardState.colors[STORE_1], activeNorth);
    renderStore(elements.southStore, boardState.board[STORE_0], "South store", boardState.colors[STORE_0], activeSouth);
  }

  function renderPitRows(displayState) {
    elements.northRow.innerHTML = "";
    elements.southRow.innerHTML = "";

    for (let action = PITS - 1; action >= 0; action -= 1) {
      elements.northRow.appendChild(createPit(1, action, displayState));
    }
    for (let action = 0; action < PITS; action += 1) {
      elements.southRow.appendChild(createPit(0, action, displayState));
    }
  }

  function createPit(player, action, displayState) {
    const index = pitIndex(player, action);
    const button = document.createElement("button");
    const legal = !busy && !animating && isHumanTurn() && player === humanPlayer && legalActions(state).includes(action);
    button.type = "button";
    button.className = `pit-button ${legal ? "legal" : "disabled"}`;
    if (displayState.activePit === index) button.classList.add("sow-active");
    if (displayState.sourcePit === index && displayState.activePit !== index) button.classList.add("sow-source");
    if (displayState.capturePit === index) button.classList.add("capture-source");
    if (displayState.sweepPits && displayState.sweepPits.includes(index)) button.classList.add("sweep");
    // The pit next to each player's bank is action 5 in the engine.
    button.style.gridColumn = String(player === 0 ? 1 : 2);
    button.style.gridRow = String(player === 0 ? action + 2 : PITS + 1 - action);
    button.disabled = !legal;
    button.setAttribute("aria-label", `${playerName(player)} pit ${action + 1}, ${displayState.board[index]} stones`);
    button.addEventListener("click", () => onPitClick(action));
    button.appendChild(stonesLayer(displayState.board[index], `pit-${player}-${action}`, displayState.colors[index]));
    button.appendChild(count(displayState.board[index]));
    return button;
  }

  function renderStore(container, stones, name, stoneColors, active) {
    container.innerHTML = "";
    container.setAttribute("aria-label", `${name}, ${stones} stones`);
    container.appendChild(stonesLayer(stones, name, stoneColors));
    container.classList.toggle("sow-active", !!active);
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
    const positions = packedStonePositions(visible, salt);
    for (let i = 0; i < visible; i += 1) {
      const stone = document.createElement("span");
      stone.className = "stone";
      stone.style.left = `${positions[i].x}%`;
      stone.style.top = `${positions[i].y}%`;
      stone.style.setProperty("--stone-color", STONE_COLORS[stoneColors[i] % STONE_COLORS.length]);
      layer.appendChild(stone);
    }
    return layer;
  }

  function packedStonePositions(total, salt) {
    // Seeded xorshift RNG — same pit + same count always gives the same layout.
    let seed = 0;
    for (let i = 0; i < salt.length; i += 1) {
      seed = (Math.imul(seed, 31) + salt.charCodeAt(i)) >>> 0;
    }
    seed = seed || 1;
    function rng() {
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      return (seed >>> 0) / 4294967296;
    }

    // Container is approx 91 × 66 px (pit after inset). Stone diameter: 14 px.
    const W = 91, H = 66, minD = 15, pad = 13;
    const positions = [];
    for (let i = 0; i < total; i += 1) {
      // Try many random candidates; keep the one with the most clearance.
      let best = null, bestDist = -1;
      const tries = Math.max(50, total * 15);
      for (let t = 0; t < tries; t += 1) {
        const x = pad + rng() * (100 - 2 * pad);
        const y = pad + rng() * (100 - 2 * pad);
        let d = Infinity;
        for (const p of positions) {
          const dx = (x - p.x) / 100 * W;
          const dy = (y - p.y) / 100 * H;
          d = Math.min(d, Math.sqrt(dx * dx + dy * dy));
        }
        if (d > bestDist) { best = { x, y }; bestDist = d; }
        if (bestDist >= minD) break; // good enough — stop early
      }
      positions.push(best);
    }
    return positions;
  }

  function statusText() {
    if (isTerminal(state)) {
      if (score(0) === score(1)) return `Draw, ${score(0)}-${score(1)}`;
      return `${score(0) > score(1) ? "South" : "North"} wins, ${score(0)}-${score(1)}`;
    }
    if (agentError) return `Agent error: ${agentError}`;
    if (animating) return `${playerName(state.currentPlayer)} moving`;
    if (busy) return `${playerName(state.currentPlayer)} thinking`;
    return `${playerName(state.currentPlayer)} to move`;
  }

  function summaryText() {
    const south = state.board.slice(0, PITS).join(" ");
    const north = state.board.slice(PITS + 1, STORE_1).join(" ");
    return `${startingStones} stones | South [${south}] | North [${north}]`;
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
    module.exports = {
      newGame,
      applyMove,
      legalActions,
      isTerminal,
      PITS,
      STONES: DEFAULT_STONES,
      DEFAULT_STONES,
      STORE_0,
      STORE_1,
    };
  }
})();
