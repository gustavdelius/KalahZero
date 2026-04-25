(() => {
  "use strict";

  const PITS = 6;
  const STORE_0 = PITS;
  const STORE_1 = 2 * PITS + 1;
  const DEFAULT_SIMULATIONS = 120;
  const C_PUCT = 1.5;

  let evaluatorPromise = null;

  self.addEventListener("message", (event) => {
    const message = event.data || {};
    if (message.type !== "move") return;
    chooseMove(message)
      .then((payload) => self.postMessage({ id: message.id, ok: true, ...payload }))
      .catch((error) => {
        self.postMessage({
          id: message.id,
          ok: false,
          error: error && error.message ? error.message : String(error),
        });
      });
  });

  async function chooseMove(message) {
    const evaluator = await loadEvaluator();
    const state = {
      board: message.board.slice(),
      currentPlayer: Number(message.currentPlayer),
    };
    if (isTerminal(state)) throw new Error("terminal position");
    const simulations = Number(message.simulations || DEFAULT_SIMULATIONS);
    const result = search(state, evaluator, simulations);
    const action = selectAction(result.visits);
    return {
      action,
      policy: result.policy,
      visits: result.visits,
      value: result.value,
      simulations,
    };
  }

  function loadEvaluator() {
    evaluatorPromise ||= fetch(new URL("residual_depth.json", self.location.href))
      .then((response) => {
        if (!response.ok) {
          throw new Error(`could not load browser agent (${response.status})`);
        }
        return response.json();
      })
      .then((payload) => new NeuralEvaluator(payload));
    return evaluatorPromise;
  }

  class NeuralEvaluator {
    constructor(payload) {
      if (payload.format !== "kalah-zero-browser-agent-v1") {
        throw new Error("unsupported browser agent format");
      }
      this.config = payload.config;
      this.tensors = {};
      for (const [name, tensor] of Object.entries(payload.state_dict)) {
        this.tensors[name] = {
          shape: tensor.shape,
          values: new Float32Array(tensor.values),
        };
      }
    }

    evaluate(state) {
      const features = encodeFeatures(state);
      const output = this.forward(features);
      const logits = output.policy;
      const legal = legalActions(state);
      const policy = maskedSoftmax(logits, legal);
      return { policy, value: output.value };
    }

    forward(input) {
      let h = relu(linear(input, this.tensors["input_layer.0.weight"], this.tensors["input_layer.0.bias"]));
      for (let block = 0; block < this.config.residual_blocks; block += 1) {
        let y = linear(
          h,
          this.tensors[`blocks.${block}.layers.0.weight`],
          this.tensors[`blocks.${block}.layers.0.bias`],
        );
        y = relu(y);
        y = linear(
          y,
          this.tensors[`blocks.${block}.layers.2.weight`],
          this.tensors[`blocks.${block}.layers.2.bias`],
        );
        const next = new Float32Array(h.length);
        for (let i = 0; i < h.length; i += 1) next[i] = Math.max(0, h[i] + y[i]);
        h = next;
      }
      const policy = linear(h, this.tensors["policy_head.weight"], this.tensors["policy_head.bias"]);
      const valueRaw = linear(h, this.tensors["value_head.0.weight"], this.tensors["value_head.0.bias"]);
      return { policy, value: Math.tanh(valueRaw[0]) };
    }
  }

  function linear(input, weight, bias) {
    const rows = weight.shape[0];
    const cols = weight.shape[1];
    const output = new Float32Array(rows);
    for (let row = 0; row < rows; row += 1) {
      let sum = bias.values[row];
      const offset = row * cols;
      for (let col = 0; col < cols; col += 1) {
        sum += weight.values[offset + col] * input[col];
      }
      output[row] = sum;
    }
    return output;
  }

  function relu(values) {
    const output = new Float32Array(values.length);
    for (let i = 0; i < values.length; i += 1) output[i] = Math.max(0, values[i]);
    return output;
  }

  function maskedSoftmax(logits, legal) {
    const output = Array(PITS).fill(0);
    if (!legal.length) return output;
    let maxLogit = -Infinity;
    for (const action of legal) maxLogit = Math.max(maxLogit, logits[action]);
    let total = 0;
    for (const action of legal) {
      const prob = Math.exp(logits[action] - maxLogit);
      output[action] = prob;
      total += prob;
    }
    if (total <= 0) {
      for (const action of legal) output[action] = 1 / legal.length;
      return output;
    }
    for (const action of legal) output[action] /= total;
    return output;
  }

  function encodeFeatures(state) {
    const player = state.currentPlayer;
    const opponent = 1 - player;
    const total = Math.max(1, state.board.reduce((sum, value) => sum + value, 0));
    const features = [];
    for (const value of pitsFor(state, player)) features.push(value / total);
    const opponentPits = pitsFor(state, opponent);
    for (let i = opponentPits.length - 1; i >= 0; i -= 1) features.push(opponentPits[i] / total);
    features.push(state.board[storeIndex(player)] / total);
    features.push(state.board[storeIndex(opponent)] / total);
    features.push(1);
    return new Float32Array(features);
  }

  function search(state, evaluator, simulations) {
    const root = createNode(state, 1, null);
    expand(root, evaluator);
    for (let i = 0; i < simulations; i += 1) {
      const path = selectPath(root);
      const leaf = path[path.length - 1];
      const value = isTerminal(leaf.state)
        ? rewardForPlayer(leaf.state, leaf.state.currentPlayer)
        : expand(leaf, evaluator).value;
      backup(path, value);
    }
    const visits = Array(PITS).fill(0);
    for (let action = 0; action < PITS; action += 1) {
      if (root.children[action]) visits[action] = root.children[action].visitCount;
    }
    const totalVisits = visits.reduce((sum, value) => sum + value, 0);
    const policy = visits.map((visit) => (totalVisits ? visit / totalVisits : 0));
    return { visits, policy, value: meanValue(root) };
  }

  function createNode(state, prior, parent) {
    return {
      state,
      prior,
      parent,
      visitCount: 0,
      valueSum: 0,
      children: Array(PITS).fill(null),
    };
  }

  function expand(node, evaluator) {
    const result = evaluator.evaluate(node.state);
    const policy = maskedPolicy(node.state, result.policy);
    for (const action of legalActions(node.state)) {
      if (!node.children[action]) {
        node.children[action] = createNode(applyMove(node.state, action), policy[action], node);
      }
    }
    return { policy, value: result.value };
  }

  function selectPath(root) {
    const path = [root];
    let node = root;
    while (isExpanded(node) && !isTerminal(node.state)) {
      node = bestChild(node);
      path.push(node);
    }
    return path;
  }

  function bestChild(parent) {
    let best = null;
    let bestScore = -Infinity;
    for (const child of parent.children) {
      if (!child) continue;
      const score = scoreChild(parent, child);
      if (score > bestScore) {
        best = child;
        bestScore = score;
      }
    }
    return best;
  }

  function scoreChild(parent, child) {
    let q = meanValue(child);
    if (child.state.currentPlayer !== parent.state.currentPlayer) q = -q;
    const exploration = C_PUCT * child.prior * Math.sqrt(parent.visitCount + 1) / (1 + child.visitCount);
    return q + exploration;
  }

  function backup(path, leafValue) {
    let valueForChild = leafValue;
    let childPlayer = path[path.length - 1].state.currentPlayer;
    for (let i = path.length - 1; i >= 0; i -= 1) {
      const node = path[i];
      const nodeValue = node.state.currentPlayer === childPlayer ? valueForChild : -valueForChild;
      node.visitCount += 1;
      node.valueSum += nodeValue;
      valueForChild = nodeValue;
      childPlayer = node.state.currentPlayer;
    }
  }

  function maskedPolicy(state, policy) {
    const legal = legalActions(state);
    const masked = Array(PITS).fill(0);
    for (const action of legal) masked[action] = Math.max(0, Number(policy[action]));
    const total = masked.reduce((sum, value) => sum + value, 0);
    if (total <= 0 && legal.length) {
      for (const action of legal) masked[action] = 1 / legal.length;
      return masked;
    }
    if (total > 0) {
      for (let i = 0; i < masked.length; i += 1) masked[i] /= total;
    }
    return masked;
  }

  function meanValue(node) {
    return node.visitCount === 0 ? 0 : node.valueSum / node.visitCount;
  }

  function isExpanded(node) {
    return node.children.some(Boolean);
  }

  function selectAction(visits) {
    let bestAction = -1;
    let bestVisits = -1;
    for (let action = 0; action < visits.length; action += 1) {
      if (visits[action] > bestVisits) {
        bestAction = action;
        bestVisits = visits[action];
      }
    }
    if (bestAction < 0 || bestVisits <= 0) throw new Error("no visited legal action");
    return bestAction;
  }

  function storeIndex(player) {
    return player === 0 ? STORE_0 : STORE_1;
  }

  function pitIndices(player) {
    return player === 0 ? [0, 1, 2, 3, 4, 5] : [7, 8, 9, 10, 11, 12];
  }

  function pitIndex(player, action) {
    return player === 0 ? action : PITS + 1 + action;
  }

  function actionForIndex(player, index) {
    return player === 0 ? index : index - (PITS + 1);
  }

  function pitsFor(state, player) {
    return pitIndices(player).map((index) => state.board[index]);
  }

  function oppositeIndex(index) {
    return 2 * PITS - index;
  }

  function sideEmpty(board, player) {
    return pitIndices(player).every((index) => board[index] === 0);
  }

  function isTerminal(state) {
    return sideEmpty(state.board, 0) || sideEmpty(state.board, 1);
  }

  function legalActions(state) {
    if (isTerminal(state)) return [];
    return pitIndices(state.currentPlayer)
      .filter((index) => state.board[index] > 0)
      .map((index) => actionForIndex(state.currentPlayer, index));
  }

  function applyMove(state, action) {
    if (!legalActions(state).includes(action)) throw new Error(`illegal action ${action}`);
    const board = state.board.slice();
    const mover = state.currentPlayer;
    const source = pitIndex(mover, action);
    let stones = board[source];
    board[source] = 0;

    const ownStore = storeIndex(mover);
    const opponentStore = storeIndex(1 - mover);
    let index = source;
    while (stones > 0) {
      index = (index + 1) % board.length;
      if (index === opponentStore) continue;
      board[index] += 1;
      stones -= 1;
    }

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
    if (sideEmpty(board, 0) || sideEmpty(board, 1)) sweepRemaining(board);
    return { board, currentPlayer: nextPlayer };
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

  function rewardForPlayer(state, player) {
    const own = state.board[storeIndex(player)];
    const other = state.board[storeIndex(1 - player)];
    if (own > other) return 1;
    if (own < other) return -1;
    return 0;
  }
})();
