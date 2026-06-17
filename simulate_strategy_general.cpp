#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using std::cerr;
using std::cin;
using std::ifstream;
using std::ofstream;
using std::runtime_error;
using std::size_t;
using std::string;
using std::uint32_t;
using std::uint64_t;
using std::vector;

struct Pos {
    int x = 0;
    int y = 0;
};

struct Rect {
    int x_min = 0;
    int y_min = 0;
    int x_max = 0;
    int y_max = 0;
};

struct Action {
    int vx = 0;
    int vy = 0;
    uint32_t id = 0;
    int eta_x = 0;
    int eta_y = 0;
};

struct ProblemInput {
    int k = 0;
    int x_init = 0;
    int y_init = 0;
    int x_max = 0;
    int y_max = 0;
    int vx_cmd_max = 0;
    int vy_cmd_max = 0;
    int eta_max = 0;
    int x_wall = 0;
    vector<long long> radius;
    vector<Rect> obstacles;
};

struct LargeState {
    vector<Pos> hist;
};

struct LargeStateHash {
    size_t operator()(const LargeState& state) const {
        size_t h = 1469598103934665603ULL;
        for (const Pos& q : state.hist) {
            h ^= static_cast<size_t>(q.x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(q.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

static bool operator==(const LargeState& a, const LargeState& b) {
    if (a.hist.size() != b.hist.size()) return false;
    for (size_t i = 0; i < a.hist.size(); ++i) {
        if (a.hist[i].x != b.hist[i].x || a.hist[i].y != b.hist[i].y) return false;
    }
    return true;
}

struct Strategy {
    uint32_t version = 0;
    uint32_t solver_code = 0;
    uint32_t k = 0;
    uint32_t history_len = 0;

    int x_max = 0;
    int y_max = 0;
    int vx_cmd_max = 0;
    int vy_cmd_max = 0;
    int eta_max = 0;
    int x_wall = 0;

    uint64_t baseP = 0;
    uint64_t action_count = 0;
    uint64_t mask_words = 0;
    vector<long long> radius;
    vector<Rect> obstacles;

    vector<uint64_t> state_ids;
    vector<LargeState> large_states;
    vector<uint64_t> masks_flat;
    std::unordered_map<uint64_t, uint32_t> state_to_index;
    std::unordered_map<LargeState, uint32_t, LargeStateHash> large_state_to_index;
};

enum class PolicyKind {
    Uniform,
    XAxisMax,
    XTargetSpeed,
};

enum class SelectionMode {
    Uniform,
    Deterministic,
    Softmax,
};

struct PolicyConfig {
    PolicyKind kind = PolicyKind::Uniform;
    SelectionMode mode = SelectionMode::Uniform;
    double target_speed = 0.0;
    double temperature = 1.0;
};

static void die(const string& msg) {
    throw runtime_error(msg);
}

static int parseIntToken(const string& text, const string& name) {
    std::istringstream in(text);
    long long value = 0;
    char extra = '\0';
    if (!(in >> value) || (in >> extra) ||
        value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        die("invalid " + name + ": " + text);
    }
    return static_cast<int>(value);
}

static int int64ToInt(int64_t value, const string& name) {
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        die(name + " is outside int range");
    }
    return static_cast<int>(value);
}

static void validateObstacle(const Rect& r, int x_max, int y_max) {
    if (r.x_min < 0 || r.x_max > x_max || r.y_min < 0 || r.y_max > y_max ||
        r.x_min > r.x_max || r.y_min > r.y_max) {
        die("invalid obstacle rectangle; expected 0 <= x_min <= x_max <= domain x_max and 0 <= y_min <= y_max <= domain y_max");
    }
}

static vector<Rect> parseObstacleTail(const vector<string>& tokens, int x_max, int y_max) {
    vector<Rect> obstacles;
    if (tokens.empty()) return obstacles;
    if (tokens[0] != "obstacles") {
        die("unknown optional simulation input section: " + tokens[0] + " ; expected obstacles");
    }
    if (tokens.size() < 2) die("obstacles section requires a count");

    int count = parseIntToken(tokens[1], "obstacle count");
    if (count < 0) die("obstacle count must be nonnegative");
    size_t expected = 2U + static_cast<size_t>(count) * 4U;
    if (tokens.size() != expected) {
        die("obstacles section has wrong number of coordinates");
    }

    obstacles.reserve(static_cast<size_t>(count));
    size_t pos = 2;
    for (int i = 0; i < count; ++i) {
        Rect r;
        r.x_min = parseIntToken(tokens[pos++], "obstacle x_min");
        r.y_min = parseIntToken(tokens[pos++], "obstacle y_min");
        r.x_max = parseIntToken(tokens[pos++], "obstacle x_max");
        r.y_max = parseIntToken(tokens[pos++], "obstacle y_max");
        validateObstacle(r, x_max, y_max);
        obstacles.push_back(r);
    }
    return obstacles;
}

static double parseDoubleToken(const string& text, const string& name) {
    std::istringstream in(text);
    double value = 0.0;
    char extra = '\0';
    if (!(in >> value) || (in >> extra)) {
        die("invalid " + name + ": " + text);
    }
    return value;
}

static SelectionMode parseSelectionMode(const string& text) {
    if (text == "deterministic") return SelectionMode::Deterministic;
    if (text == "softmax") return SelectionMode::Softmax;
    die("unknown policy selection mode: " + text + " ; expected deterministic or softmax");
    return SelectionMode::Uniform;
}

static PolicyConfig parsePolicyConfig(const vector<string>& tokens) {
    PolicyConfig policy;
    if (tokens.empty()) return policy;

    const string& kind = tokens[0];
    if (kind == "uniform") {
        if (tokens.size() != 1) die("uniform policy takes no extra parameters");
        return policy;
    }

    if (kind == "x_axis_max") {
        if (tokens.size() < 2) die("x_axis_max policy requires a selection mode");
        policy.kind = PolicyKind::XAxisMax;
        policy.mode = parseSelectionMode(tokens[1]);
        if (policy.mode == SelectionMode::Deterministic) {
            if (tokens.size() != 2) die("x_axis_max deterministic takes no extra parameters");
        } else if (policy.mode == SelectionMode::Softmax) {
            if (tokens.size() != 3) die("x_axis_max softmax requires exactly one temperature parameter");
            policy.temperature = parseDoubleToken(tokens[2], "temperature");
            if (!(policy.temperature > 0.0)) die("temperature must be positive");
        }
        return policy;
    }

    if (kind == "x_target_speed") {
        if (tokens.size() < 3) die("x_target_speed policy requires a selection mode and target speed");
        policy.kind = PolicyKind::XTargetSpeed;
        policy.mode = parseSelectionMode(tokens[1]);
        policy.target_speed = parseDoubleToken(tokens[2], "target_speed");
        if (!(policy.target_speed >= 0.0)) die("target_speed must be nonnegative");
        if (policy.mode == SelectionMode::Deterministic) {
            if (tokens.size() != 3) die("x_target_speed deterministic requires exactly one target speed parameter");
        } else if (policy.mode == SelectionMode::Softmax) {
            if (tokens.size() != 4) die("x_target_speed softmax requires target speed and temperature parameters");
            policy.temperature = parseDoubleToken(tokens[3], "temperature");
            if (!(policy.temperature > 0.0)) die("temperature must be positive");
        }
        return policy;
    }

    die("unknown simulation policy: " + kind);
    return policy;
}

template <typename T>
static void readBinary(ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) die("failed while reading strategy file");
}

static uint64_t checkedMul(uint64_t a, uint64_t b, const string& what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        die("uint64 overflow while computing " + what);
    }
    return a * b;
}

static vector<vector<long long>> computeBinom(int k) {
    vector<vector<long long>> c(static_cast<size_t>(k) + 1, vector<long long>(static_cast<size_t>(k) + 1, 0));
    for (int n = 0; n <= k; ++n) {
        c[static_cast<size_t>(n)][0] = 1;
        c[static_cast<size_t>(n)][static_cast<size_t>(n)] = 1;
        for (int i = 1; i < n; ++i) {
            __int128 val = static_cast<__int128>(c[static_cast<size_t>(n - 1)][static_cast<size_t>(i - 1)])
                         + static_cast<__int128>(c[static_cast<size_t>(n - 1)][static_cast<size_t>(i)]);
            if (val > std::numeric_limits<long long>::max()) die("binomial coefficient overflow; k is too large");
            c[static_cast<size_t>(n)][static_cast<size_t>(i)] = static_cast<long long>(val);
        }
    }
    return c;
}

static vector<Action> makeActions(const ProblemInput& p) {
    uint64_t action_count = checkedMul(static_cast<uint64_t>(p.vx_cmd_max) + 1ULL,
                                       static_cast<uint64_t>(p.vy_cmd_max) + 1ULL,
                                       "number of actions");
    if (action_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        die("too many actions for uint32_t ids");
    }
    vector<Action> actions;
    actions.reserve(static_cast<size_t>(action_count));
    uint32_t aid = 0;
    for (int vx = 0; vx <= p.vx_cmd_max; ++vx) {
        for (int vy = 0; vy <= p.vy_cmd_max; ++vy) {
            Action a;
            a.vx = vx;
            a.vy = vy;
            a.id = aid++;
            a.eta_x = std::min(p.eta_max, vx);
            a.eta_y = std::min(p.eta_max, vy);
            actions.push_back(a);
        }
    }
    return actions;
}

static inline uint64_t posId(const ProblemInput& p, const Pos& q) {
    return static_cast<uint64_t>(q.x) * (static_cast<uint64_t>(p.y_max) + 1ULL) + static_cast<uint64_t>(q.y);
}

static inline uint64_t encodePrefix(const ProblemInput& p,
                                    const vector<Pos>& hist,
                                    int history_len) {
    uint64_t baseP = checkedMul(static_cast<uint64_t>(p.x_max) + 1ULL,
                                static_cast<uint64_t>(p.y_max) + 1ULL,
                                "number of positions");
    uint64_t id = 0;
    uint64_t mult = 1;
    for (int i = 0; i < history_len; ++i) {
        uint64_t term = checkedMul(posId(p, hist[static_cast<size_t>(i)]), mult, "state id term");
        if (term > std::numeric_limits<uint64_t>::max() - id) die("state id overflow");
        id += term;
        if (i + 1 < history_len) mult = checkedMul(mult, baseP, "state id multiplier");
    }
    return id;
}

static LargeState largePrefix(const vector<Pos>& hist, int history_len) {
    LargeState state;
    state.hist.reserve(static_cast<size_t>(history_len));
    for (int i = 0; i < history_len; ++i) {
        state.hist.push_back(hist[static_cast<size_t>(i)]);
    }
    return state;
}

static string positionViolationReason(const ProblemInput& p, const Pos& q) {
    if (q.x < 0 || q.x > p.x_max || q.y < 0 || q.y > p.y_max) {
        return "position_violation";
    }
    if (q.x > p.x_wall) {
        return "wall_violation";
    }
    for (const Rect& r : p.obstacles) {
        if (q.x >= r.x_min && q.x <= r.x_max &&
            q.y >= r.y_min && q.y <= r.y_max) {
            return "obstacle_violation";
        }
    }
    return "ok";
}

static inline bool positionOK(const ProblemInput& p, const Pos& q) {
    return positionViolationReason(p, q) == "ok";
}

static vector<std::pair<long long,long long>> derivativesAtNext(
        const ProblemInput& p,
        const vector<vector<long long>>& binom,
        const vector<Pos>& hist,
        const Pos& next) {
    vector<std::pair<long long,long long>> vals;
    vals.reserve(static_cast<size_t>(p.k));

    for (int d = 1; d <= p.k; ++d) {
        __int128 sx = next.x;
        __int128 sy = next.y;
        for (int i = 1; i <= d; ++i) {
            long long c = binom[static_cast<size_t>(d)][static_cast<size_t>(i)];
            if (i & 1) c = -c;
            sx += static_cast<__int128>(c) * hist[static_cast<size_t>(i - 1)].x;
            sy += static_cast<__int128>(c) * hist[static_cast<size_t>(i - 1)].y;
        }
        if (sx < std::numeric_limits<long long>::min() || sx > std::numeric_limits<long long>::max() ||
            sy < std::numeric_limits<long long>::min() || sy > std::numeric_limits<long long>::max()) {
            die("derivative value overflow");
        }
        vals.emplace_back(static_cast<long long>(sx), static_cast<long long>(sy));
    }
    return vals;
}

static bool transitionOK(const ProblemInput& p,
                         const vector<vector<long long>>& binom,
                         const vector<Pos>& hist,
                         const Pos& next,
                         string& reason) {
    if (!positionOK(p, next)) {
        reason = positionViolationReason(p, next);
        return false;
    }
    auto vals = derivativesAtNext(p, binom, hist, next);
    for (int d = 1; d <= p.k; ++d) {
        __int128 dx = vals[static_cast<size_t>(d - 1)].first;
        __int128 dy = vals[static_cast<size_t>(d - 1)].second;
        __int128 lhs = dx * dx + dy * dy;
        __int128 r = p.radius[static_cast<size_t>(d - 1)];
        __int128 rhs = r * r;
        if (lhs > rhs) {
            reason = "D" + std::to_string(d) + "_violation";
            return false;
        }
    }
    reason = "ok";
    return true;
}

static bool maskEmpty(const vector<uint64_t>& masks_flat,
                      uint64_t mask_words,
                      uint32_t index) {
    size_t off = static_cast<size_t>(index) * static_cast<size_t>(mask_words);
    for (uint64_t w = 0; w < mask_words; ++w) {
        if (masks_flat[off + static_cast<size_t>(w)] != 0ULL) return false;
    }
    return true;
}

static vector<uint32_t> allowedActionIds(const Strategy& s, uint32_t state_index) {
    size_t off = static_cast<size_t>(state_index) * static_cast<size_t>(s.mask_words);
    vector<uint32_t> ids;
    for (uint64_t w = 0; w < s.mask_words; ++w) {
        uint64_t word = s.masks_flat[off + static_cast<size_t>(w)];
        while (word != 0ULL) {
            uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(word));
            uint64_t aid = w * 64ULL + bit;
            if (aid >= s.action_count) die("strategy mask contains invalid action id");
            ids.push_back(static_cast<uint32_t>(aid));
            word &= (word - 1ULL);
        }
    }
    return ids;
}

static uint32_t sampleUniformActionId(const vector<uint32_t>& action_ids,
                                      std::mt19937_64& rng) {
    if (action_ids.empty()) die("attempted to sample from an empty action set");
    std::uniform_int_distribution<uint64_t> dist(0ULL, static_cast<uint64_t>(action_ids.size() - 1));
    return action_ids[static_cast<size_t>(dist(rng))];
}

static uint32_t sampleSoftmaxActionId(const vector<uint32_t>& action_ids,
                                      const vector<double>& scores,
                                      double temperature,
                                      std::mt19937_64& rng) {
    if (action_ids.empty()) die("attempted to sample from an empty action set");
    if (action_ids.size() != scores.size()) die("internal softmax score/action mismatch");

    double max_score = scores[0];
    for (double score : scores) {
        max_score = std::max(max_score, score);
    }

    vector<double> weights(action_ids.size(), 0.0);
    double total = 0.0;
    for (size_t i = 0; i < action_ids.size(); ++i) {
        double weight = std::exp((scores[i] - max_score) / temperature);
        weights[i] = weight;
        total += weight;
    }
    if (!(total > 0.0)) die("softmax produced no positive weights");

    std::uniform_real_distribution<double> dist(0.0, total);
    double r = dist(rng);
    for (size_t i = 0; i < action_ids.size(); ++i) {
        if (r <= weights[i]) return action_ids[i];
        r -= weights[i];
    }
    return action_ids.back();
}

static double commandSpeed(const Action& a) {
    return std::sqrt(static_cast<double>(a.vx) * static_cast<double>(a.vx) +
                     static_cast<double>(a.vy) * static_cast<double>(a.vy));
}

static uint32_t selectXAxisMaxActionId(const vector<uint32_t>& allowed_ids,
                                       const vector<Action>& actions,
                                       const PolicyConfig& policy,
                                       std::mt19937_64& rng) {
    vector<uint32_t> candidates;
    for (uint32_t aid : allowed_ids) {
        if (actions[aid].vy == 0) candidates.push_back(aid);
    }
    if (candidates.empty()) return sampleUniformActionId(allowed_ids, rng);

    if (policy.mode == SelectionMode::Deterministic) {
        uint32_t best = candidates[0];
        for (uint32_t aid : candidates) {
            if (actions[aid].vx > actions[best].vx ||
                (actions[aid].vx == actions[best].vx && aid < best)) {
                best = aid;
            }
        }
        return best;
    }

    vector<double> scores;
    scores.reserve(candidates.size());
    for (uint32_t aid : candidates) {
        scores.push_back(static_cast<double>(actions[aid].vx));
    }
    return sampleSoftmaxActionId(candidates, scores, policy.temperature, rng);
}

static uint32_t selectXTargetSpeedActionId(const vector<uint32_t>& allowed_ids,
                                           const vector<Action>& actions,
                                           const PolicyConfig& policy,
                                           std::mt19937_64& rng) {
    if (policy.mode == SelectionMode::Deterministic) {
        uint32_t best = allowed_ids[0];
        for (uint32_t aid : allowed_ids) {
            const Action& a = actions[aid];
            const Action& b = actions[best];
            double a_diff = std::abs(commandSpeed(a) - policy.target_speed);
            double b_diff = std::abs(commandSpeed(b) - policy.target_speed);
            if (a.vx > b.vx ||
                (a.vx == b.vx && a_diff < b_diff) ||
                (a.vx == b.vx && a_diff == b_diff && a.vy < b.vy) ||
                (a.vx == b.vx && a_diff == b_diff && a.vy == b.vy && aid < best)) {
                best = aid;
            }
        }
        return best;
    }

    int max_vx = actions[allowed_ids[0]].vx;
    for (uint32_t aid : allowed_ids) {
        max_vx = std::max(max_vx, actions[aid].vx);
    }

    vector<uint32_t> candidates;
    vector<double> scores;
    for (uint32_t aid : allowed_ids) {
        const Action& a = actions[aid];
        if (a.vx != max_vx) continue;
        candidates.push_back(aid);
        scores.push_back(-std::abs(commandSpeed(a) - policy.target_speed));
    }
    return sampleSoftmaxActionId(candidates, scores, policy.temperature, rng);
}

static uint32_t selectActionId(const Strategy& s,
                               uint32_t state_index,
                               const vector<Action>& actions,
                               const PolicyConfig& policy,
                               std::mt19937_64& rng) {
    vector<uint32_t> allowed_ids = allowedActionIds(s, state_index);
    if (allowed_ids.empty()) die("attempted to select from an empty action mask");

    if (policy.kind == PolicyKind::Uniform) {
        return sampleUniformActionId(allowed_ids, rng);
    }
    if (policy.kind == PolicyKind::XAxisMax) {
        return selectXAxisMaxActionId(allowed_ids, actions, policy, rng);
    }
    if (policy.kind == PolicyKind::XTargetSpeed) {
        return selectXTargetSpeedActionId(allowed_ids, actions, policy, rng);
    }

    die("unknown policy kind");
    return 0U;
}

static Strategy readStrategy(const string& path) {
    ifstream in(path, std::ios::binary);
    if (!in) die("could not open strategy file: " + path);

    char magic[8];
    in.read(magic, 8);
    if (!in) die("failed to read strategy magic");
    const char expected[8] = {'G','S','T','R','A','T','K','2'};
    for (int i = 0; i < 8; ++i) {
        if (magic[i] != expected[i]) die("strategy file has wrong magic header; expected GSTRATK2");
    }

    Strategy s;
    readBinary(in, s.version);
    if (s.version != 2U && s.version != 3U && s.version != 4U) {
        die("strategy file has unsupported version");
    }
    readBinary(in, s.solver_code);
    readBinary(in, s.k);
    readBinary(in, s.history_len);

    int64_t tmp = 0;
    readBinary(in, tmp); s.x_max = static_cast<int>(tmp);
    readBinary(in, tmp); s.y_max = static_cast<int>(tmp);
    readBinary(in, tmp); s.vx_cmd_max = static_cast<int>(tmp);
    readBinary(in, tmp); s.vy_cmd_max = static_cast<int>(tmp);
    readBinary(in, tmp); s.eta_max = static_cast<int>(tmp);
    readBinary(in, tmp); s.x_wall = static_cast<int>(tmp);

    readBinary(in, s.baseP);
    readBinary(in, s.action_count);
    readBinary(in, s.mask_words);

    s.radius.resize(static_cast<size_t>(s.k));
    for (uint32_t d = 0; d < s.k; ++d) {
        int64_t r = 0;
        readBinary(in, r);
        s.radius[static_cast<size_t>(d)] = static_cast<long long>(r);
    }

    if (s.version >= 3U) {
        uint64_t obstacle_count = 0;
        readBinary(in, obstacle_count);
        if (obstacle_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            die("too many obstacles in strategy file");
        }
        s.obstacles.resize(static_cast<size_t>(obstacle_count));
        for (uint64_t i = 0; i < obstacle_count; ++i) {
            int64_t tmp_rect = 0;
            Rect r;
            readBinary(in, tmp_rect); r.x_min = int64ToInt(tmp_rect, "obstacle x_min");
            readBinary(in, tmp_rect); r.y_min = int64ToInt(tmp_rect, "obstacle y_min");
            readBinary(in, tmp_rect); r.x_max = int64ToInt(tmp_rect, "obstacle x_max");
            readBinary(in, tmp_rect); r.y_max = int64ToInt(tmp_rect, "obstacle y_max");
            s.obstacles[static_cast<size_t>(i)] = r;
        }
    }

    uint64_t nwin = 0;
    readBinary(in, nwin);
    if (nwin > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        die("too many winning states for uint32_t map indices in simulator");
    }

    s.masks_flat.resize(static_cast<size_t>(nwin) * static_cast<size_t>(s.mask_words));
    if (s.version == 4U) {
        s.large_states.resize(static_cast<size_t>(nwin));
        s.large_state_to_index.reserve(static_cast<size_t>(nwin) * 2 + 1);
        s.large_state_to_index.max_load_factor(0.70f);
    } else {
        s.state_ids.resize(static_cast<size_t>(nwin));
        s.state_to_index.reserve(static_cast<size_t>(nwin) * 2 + 1);
        s.state_to_index.max_load_factor(0.70f);
    }

    for (uint64_t i = 0; i < nwin; ++i) {
        if (s.version == 4U) {
            LargeState state;
            state.hist.resize(static_cast<size_t>(s.history_len));
            for (uint32_t j = 0; j < s.history_len; ++j) {
                int64_t qx = 0;
                int64_t qy = 0;
                readBinary(in, qx);
                readBinary(in, qy);
                state.hist[static_cast<size_t>(j)].x = int64ToInt(qx, "state x");
                state.hist[static_cast<size_t>(j)].y = int64ToInt(qy, "state y");
            }
            s.large_states[static_cast<size_t>(i)] = state;
            s.large_state_to_index.emplace(state, static_cast<uint32_t>(i));
        } else {
            uint64_t sid = 0;
            readBinary(in, sid);
            s.state_ids[static_cast<size_t>(i)] = sid;
            s.state_to_index.emplace(sid, static_cast<uint32_t>(i));
        }
        size_t off = static_cast<size_t>(i) * static_cast<size_t>(s.mask_words);
        for (uint64_t w = 0; w < s.mask_words; ++w) {
            readBinary(in, s.masks_flat[off + static_cast<size_t>(w)]);
        }
    }

    return s;
}

static void validateStrategy(const ProblemInput& p, const Strategy& s) {
    auto fail = [](const string& msg) { die("strategy/input mismatch: " + msg); };
    if (s.version != 2U && s.version != 3U && s.version != 4U) fail("unsupported strategy version");
    if (s.version == 4U && s.solver_code != 0U) fail("version 4 is only valid for basic solver strategies");
    if (s.version == 4U && s.k < 5U) fail("version 4 is only valid for k >= 5");
    if (s.k != static_cast<uint32_t>(p.k)) fail("k differs");
    if (s.x_max != p.x_max) fail("x_max differs");
    if (s.y_max != p.y_max) fail("y_max differs");
    if (s.vx_cmd_max != p.vx_cmd_max) fail("vx_cmd_max differs");
    if (s.vy_cmd_max != p.vy_cmd_max) fail("vy_cmd_max differs");
    if (s.eta_max != p.eta_max) fail("eta_max differs");
    if (s.x_wall != p.x_wall) fail("x_wall differs");
    if (s.radius.size() != p.radius.size()) fail("radius vector length differs");
    for (size_t i = 0; i < p.radius.size(); ++i) {
        if (s.radius[i] != p.radius[i]) fail("one or more derivative radii differ");
    }
    if (s.obstacles.size() != p.obstacles.size()) fail("obstacle count differs");
    for (size_t i = 0; i < p.obstacles.size(); ++i) {
        const Rect& a = s.obstacles[i];
        const Rect& b = p.obstacles[i];
        validateObstacle(a, p.x_max, p.y_max);
        if (a.x_min != b.x_min || a.y_min != b.y_min ||
            a.x_max != b.x_max || a.y_max != b.y_max) {
            fail("one or more obstacle rectangles differ");
        }
    }

    uint64_t expected_baseP = checkedMul(static_cast<uint64_t>(p.x_max) + 1ULL,
                                         static_cast<uint64_t>(p.y_max) + 1ULL,
                                         "number of positions");
    uint64_t expected_actions = checkedMul(static_cast<uint64_t>(p.vx_cmd_max) + 1ULL,
                                           static_cast<uint64_t>(p.vy_cmd_max) + 1ULL,
                                           "number of actions");
    uint64_t expected_words = (expected_actions + 63ULL) / 64ULL;
    if (s.baseP != expected_baseP) fail("baseP differs");
    if (s.action_count != expected_actions) fail("action count differs");
    if (s.mask_words != expected_words) fail("mask word count differs");
    if (s.history_len == 0U) fail("strategy history length is zero");
    if (s.version == 4U && s.history_len != s.k + 1U) {
        fail("version 4 basic strategy must use history length k + 1");
    }
}

int main() {
    try {
        ProblemInput p;
        string strategy_path;
        string output_csv_path;
        uint64_t num_traces = 0;
        uint64_t max_steps = 0;
        uint64_t seed = 0;

        cin >> p.k;
        cin >> p.x_init >> p.y_init;
        cin >> p.x_max >> p.y_max;
        cin >> p.vx_cmd_max >> p.vy_cmd_max;
        cin >> p.eta_max;
        cin >> p.x_wall;
        if (!cin) die("failed to read problem parameters");

        p.radius.resize(static_cast<size_t>(p.k));
        for (int i = 0; i < p.k; ++i) {
            cin >> p.radius[static_cast<size_t>(i)];
            if (p.radius[static_cast<size_t>(i)] < 0) die("derivative radii must be nonnegative");
        }

        cin >> strategy_path;
        cin >> num_traces;
        cin >> max_steps;
        cin >> output_csv_path;
        cin >> seed;
        if (!cin) die("failed to read strategy path, trace settings, output path, or seed");

        vector<string> optional_tokens;
        string token;
        while (cin >> token) {
            optional_tokens.push_back(token);
        }
        if (cin.bad()) die("failed while reading optional policy parameters");

        auto obstacle_it = std::find(optional_tokens.begin(), optional_tokens.end(), "obstacles");
        vector<string> policy_tokens(optional_tokens.begin(), obstacle_it);
        vector<string> obstacle_tokens;
        if (obstacle_it != optional_tokens.end()) {
            obstacle_tokens.assign(obstacle_it, optional_tokens.end());
        }
        PolicyConfig policy = parsePolicyConfig(policy_tokens);
        p.obstacles = parseObstacleTail(obstacle_tokens, p.x_max, p.y_max);
        Pos initial_pos{p.x_init, p.y_init};
        if (!positionOK(p, initial_pos)) {
            die("initial position is unsafe: " + positionViolationReason(p, initial_pos));
        }

        Strategy s = readStrategy(strategy_path);
        validateStrategy(p, s);

        vector<Action> actions = makeActions(p);
        vector<vector<long long>> binom = computeBinom(p.k);
        int full_history_len = std::max<int>(static_cast<int>(s.history_len), p.k + 1);

        ofstream csv(output_csv_path);
        if (!csv) die("could not open output CSV: " + output_csv_path);

        csv << "trace,step,x,y,cmd_vx,cmd_vy,noise_x,noise_y,real_dx,real_dy";
        for (int d = 1; d <= p.k; ++d) {
            csv << ",D" << d << "_x,D" << d << "_y";
        }
        csv << ",safe,reason\n";

        std::mt19937_64 rng(seed);

        for (uint64_t tr = 0; tr < num_traces; ++tr) {
            vector<Pos> hist(static_cast<size_t>(full_history_len), initial_pos);

            for (uint64_t step = 1; step <= max_steps; ++step) {
                bool found_state = false;
                uint32_t state_index = 0;
                if (s.version == 4U) {
                    LargeState sid = largePrefix(hist, static_cast<int>(s.history_len));
                    auto it = s.large_state_to_index.find(sid);
                    if (it != s.large_state_to_index.end()) {
                        found_state = true;
                        state_index = it->second;
                    }
                } else {
                    uint64_t sid = encodePrefix(p, hist, static_cast<int>(s.history_len));
                    auto it = s.state_to_index.find(sid);
                    if (it != s.state_to_index.end()) {
                        found_state = true;
                        state_index = it->second;
                    }
                }

                if (!found_state || maskEmpty(s.masks_flat, s.mask_words, state_index)) {
                    csv << tr << ',' << step << ',' << hist[0].x << ',' << hist[0].y
                        << ",-1,-1,0,0,0,0";
                    for (int d = 1; d <= p.k; ++d) csv << ",0,0";
                    csv << ",0,no_allowed_action\n";
                    break;
                }

                uint32_t aid = selectActionId(s, state_index, actions, policy, rng);
                const Action& a = actions[aid];

                std::uniform_int_distribution<int> ndx(-a.eta_x, a.eta_x);
                std::uniform_int_distribution<int> ndy(-a.eta_y, a.eta_y);
                int noise_x = ndx(rng);
                int noise_y = ndy(rng);

                Pos cur = hist[0];
                Pos next;
                next.x = cur.x + a.vx + noise_x;
                next.y = cur.y + a.vy + noise_y;

                vector<std::pair<long long,long long>> derivs = derivativesAtNext(p, binom, hist, next);
                string reason;
                bool safe = transitionOK(p, binom, hist, next, reason);

                csv << tr << ',' << step << ',' << next.x << ',' << next.y
                    << ',' << a.vx << ',' << a.vy
                    << ',' << noise_x << ',' << noise_y
                    << ',' << (next.x - cur.x) << ',' << (next.y - cur.y);
                for (int d = 1; d <= p.k; ++d) {
                    csv << ',' << derivs[static_cast<size_t>(d - 1)].first
                        << ',' << derivs[static_cast<size_t>(d - 1)].second;
                }
                csv << ',' << (safe ? 1 : 0) << ',' << reason << '\n';

                if (!safe) break;

                for (int i = full_history_len - 1; i >= 1; --i) {
                    hist[static_cast<size_t>(i)] = hist[static_cast<size_t>(i - 1)];
                }
                hist[0] = next;
            }
        }

        cerr << "wrote traces to: " << output_csv_path << "\n";
        size_t loaded_states = (s.version == 4U) ? s.large_states.size() : s.state_ids.size();
        cerr << "loaded winning states: " << loaded_states << "\n";
        cerr << "strategy history length: " << s.history_len << "\n";
        cerr << "action count: " << s.action_count << ", mask words per state: " << s.mask_words << "\n";
        return 0;
    } catch (const std::exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
