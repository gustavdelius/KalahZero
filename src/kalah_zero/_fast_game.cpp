#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int PLAYER_0 = 0;
constexpr int PLAYER_1 = 1;

std::vector<int> board_from_object(PyObject* object) {
    PyObject* sequence = PySequence_Fast(object, "board must be a sequence");
    if (sequence == nullptr) {
        throw std::runtime_error("board must be a sequence");
    }
    Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    std::vector<int> board(static_cast<size_t>(size));
    PyObject** items = PySequence_Fast_ITEMS(sequence);
    for (Py_ssize_t i = 0; i < size; ++i) {
        long value = PyLong_AsLong(items[i]);
        if (PyErr_Occurred()) {
            Py_DECREF(sequence);
            throw std::runtime_error("board entries must be integers");
        }
        board[static_cast<size_t>(i)] = static_cast<int>(value);
    }
    Py_DECREF(sequence);
    return board;
}

PyObject* tuple_from_board(const std::vector<int>& board) {
    PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(board.size()));
    if (tuple == nullptr) {
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(board.size()); ++i) {
        PyObject* value = PyLong_FromLong(board[static_cast<size_t>(i)]);
        if (value == nullptr) {
            Py_DECREF(tuple);
            return nullptr;
        }
        PyTuple_SET_ITEM(tuple, i, value);
    }
    return tuple;
}

void validate_player(int player) {
    if (player != PLAYER_0 && player != PLAYER_1) {
        throw std::invalid_argument("player must be 0 or 1");
    }
}

int store_index(int player, int pits) {
    return player == PLAYER_0 ? pits : 2 * pits + 1;
}

int pit_index(int player, int action, int pits) {
    if (action < 0 || action >= pits) {
        throw std::invalid_argument("illegal action");
    }
    return player == PLAYER_0 ? action : pits + 1 + action;
}

bool side_empty(const std::vector<int>& board, int player, int pits) {
    int start = player == PLAYER_0 ? 0 : pits + 1;
    for (int index = start; index < start + pits; ++index) {
        if (board[static_cast<size_t>(index)] != 0) {
            return false;
        }
    }
    return true;
}

bool is_own_pit(int index, int player, int pits) {
    if (player == PLAYER_0) {
        return 0 <= index && index < pits;
    }
    return pits < index && index < 2 * pits + 1;
}

void sweep_remaining(std::vector<int>& board, int pits) {
    int store_0 = store_index(PLAYER_0, pits);
    int store_1 = store_index(PLAYER_1, pits);
    for (int index = 0; index < pits; ++index) {
        board[static_cast<size_t>(store_0)] += board[static_cast<size_t>(index)];
        board[static_cast<size_t>(index)] = 0;
    }
    for (int index = pits + 1; index < 2 * pits + 1; ++index) {
        board[static_cast<size_t>(store_1)] += board[static_cast<size_t>(index)];
        board[static_cast<size_t>(index)] = 0;
    }
}

void validate_board(const std::vector<int>& board, int pits) {
    if (pits <= 0) {
        throw std::invalid_argument("pits must be positive");
    }
    if (static_cast<int>(board.size()) != 2 * pits + 2) {
        throw std::invalid_argument("board length does not match pits");
    }
}

PyObject* py_is_terminal(PyObject*, PyObject* args) {
    PyObject* board_object = nullptr;
    int pits = 0;
    if (!PyArg_ParseTuple(args, "Oi", &board_object, &pits)) {
        return nullptr;
    }
    try {
        std::vector<int> board = board_from_object(board_object);
        validate_board(board, pits);
        if (side_empty(board, PLAYER_0, pits) || side_empty(board, PLAYER_1, pits)) {
            Py_RETURN_TRUE;
        }
        Py_RETURN_FALSE;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_legal_actions(PyObject*, PyObject* args) {
    PyObject* board_object = nullptr;
    int current_player = 0;
    int pits = 0;
    if (!PyArg_ParseTuple(args, "Oii", &board_object, &current_player, &pits)) {
        return nullptr;
    }
    try {
        validate_player(current_player);
        std::vector<int> board = board_from_object(board_object);
        validate_board(board, pits);
        PyObject* actions = PyList_New(0);
        if (actions == nullptr) {
            return nullptr;
        }
        if (side_empty(board, PLAYER_0, pits) || side_empty(board, PLAYER_1, pits)) {
            return actions;
        }
        int start = current_player == PLAYER_0 ? 0 : pits + 1;
        for (int index = start; index < start + pits; ++index) {
            if (board[static_cast<size_t>(index)] > 0) {
                PyObject* action = PyLong_FromLong(index - start);
                if (action == nullptr || PyList_Append(actions, action) < 0) {
                    Py_XDECREF(action);
                    Py_DECREF(actions);
                    return nullptr;
                }
                Py_DECREF(action);
            }
        }
        return actions;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_apply(PyObject*, PyObject* args) {
    PyObject* board_object = nullptr;
    int current_player = 0;
    int pits = 0;
    int action = 0;
    if (!PyArg_ParseTuple(args, "Oiii", &board_object, &current_player, &pits, &action)) {
        return nullptr;
    }
    try {
        validate_player(current_player);
        std::vector<int> board = board_from_object(board_object);
        validate_board(board, pits);
        if (side_empty(board, PLAYER_0, pits) || side_empty(board, PLAYER_1, pits)) {
            throw std::invalid_argument("cannot apply an action to a terminal state");
        }
        int source = pit_index(current_player, action, pits);
        if (board[static_cast<size_t>(source)] <= 0) {
            throw std::invalid_argument("illegal action");
        }

        int stones = board[static_cast<size_t>(source)];
        board[static_cast<size_t>(source)] = 0;
        int own_store = store_index(current_player, pits);
        int opponent_store = store_index(1 - current_player, pits);
        int index = source;
        int board_size = static_cast<int>(board.size());
        while (stones > 0) {
            index = (index + 1) % board_size;
            if (index == opponent_store) {
                continue;
            }
            board[static_cast<size_t>(index)] += 1;
            stones -= 1;
        }

        bool captured_stones = false;
        if (is_own_pit(index, current_player, pits) && board[static_cast<size_t>(index)] == 1) {
            int opposite = 2 * pits - index;
            int captured = board[static_cast<size_t>(opposite)];
            if (captured > 0) {
                board[static_cast<size_t>(opposite)] = 0;
                board[static_cast<size_t>(index)] = 0;
                board[static_cast<size_t>(own_store)] += captured + 1;
                captured_stones = true;
            }
        }

        int next_player = (index == own_store || captured_stones) ? current_player : 1 - current_player;
        if (side_empty(board, PLAYER_0, pits) || side_empty(board, PLAYER_1, pits)) {
            sweep_remaining(board, pits);
        }

        PyObject* board_tuple = tuple_from_board(board);
        if (board_tuple == nullptr) {
            return nullptr;
        }
        PyObject* result = Py_BuildValue("Ni", board_tuple, next_player);
        return result;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_store_for(PyObject*, PyObject* args) {
    PyObject* board_object = nullptr;
    int pits = 0;
    int player = 0;
    if (!PyArg_ParseTuple(args, "Oii", &board_object, &pits, &player)) {
        return nullptr;
    }
    try {
        validate_player(player);
        std::vector<int> board = board_from_object(board_object);
        validate_board(board, pits);
        return PyLong_FromLong(board[static_cast<size_t>(store_index(player, pits))]);
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_pits_for(PyObject*, PyObject* args) {
    PyObject* board_object = nullptr;
    int pits = 0;
    int player = 0;
    if (!PyArg_ParseTuple(args, "Oii", &board_object, &pits, &player)) {
        return nullptr;
    }
    try {
        validate_player(player);
        std::vector<int> board = board_from_object(board_object);
        validate_board(board, pits);
        int start = player == PLAYER_0 ? 0 : pits + 1;
        PyObject* tuple = PyTuple_New(pits);
        if (tuple == nullptr) {
            return nullptr;
        }
        for (int action = 0; action < pits; ++action) {
            PyObject* value = PyLong_FromLong(board[static_cast<size_t>(start + action)]);
            if (value == nullptr) {
                Py_DECREF(tuple);
                return nullptr;
            }
            PyTuple_SET_ITEM(tuple, action, value);
        }
        return tuple;
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_reward_for_player(PyObject*, PyObject* args) {
    PyObject* board_object = nullptr;
    int pits = 0;
    int player = 0;
    if (!PyArg_ParseTuple(args, "Oii", &board_object, &pits, &player)) {
        return nullptr;
    }
    try {
        validate_player(player);
        std::vector<int> board = board_from_object(board_object);
        validate_board(board, pits);
        int own = board[static_cast<size_t>(store_index(player, pits))];
        int other = board[static_cast<size_t>(store_index(1 - player, pits))];
        double reward = own > other ? 1.0 : own < other ? -1.0 : 0.0;
        return PyFloat_FromDouble(reward);
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyObject* py_normalized_store_margin(PyObject*, PyObject* args) {
    PyObject* board_object = nullptr;
    int pits = 0;
    int player = 0;
    if (!PyArg_ParseTuple(args, "Oii", &board_object, &pits, &player)) {
        return nullptr;
    }
    try {
        validate_player(player);
        std::vector<int> board = board_from_object(board_object);
        validate_board(board, pits);
        int own = board[static_cast<size_t>(store_index(player, pits))];
        int other = board[static_cast<size_t>(store_index(1 - player, pits))];
        int total = 0;
        for (int stones : board) {
            total += stones;
        }
        if (total <= 0) {
            total = 1;
        }
        return PyFloat_FromDouble(static_cast<double>(own - other) / static_cast<double>(total));
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"is_terminal", py_is_terminal, METH_VARARGS, "Return whether either side is empty."},
    {"legal_actions", py_legal_actions, METH_VARARGS, "Return legal local pit actions."},
    {"apply", py_apply, METH_VARARGS, "Apply a move and return `(board, next_player)`."},
    {"store_for", py_store_for, METH_VARARGS, "Return a player's store count."},
    {"pits_for", py_pits_for, METH_VARARGS, "Return a player's pit counts."},
    {"reward_for_player", py_reward_for_player, METH_VARARGS, "Return terminal score sign."},
    {"normalized_store_margin", py_normalized_store_margin, METH_VARARGS, "Return normalized store margin."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_fast_game",
    "C++ acceleration helpers for Kalah state transitions.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__fast_game() {
    return PyModule_Create(&module);
}
