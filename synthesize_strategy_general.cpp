#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/resource.h>
#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

using std::cerr;
using std::cin;
using std::cout;
using std::deque;
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

struct Params {
    int k = 0;
    string solver_mode;
    uint32_t solver_code = 0;

    int x_max = 0;
    int y_max = 0;
    int vx_cmd_max = 0;
    int vy_cmd_max = 0;
    int eta_max = 0;
    int x_wall = 0;

    vector<long long> radius;              // radius[d-1] for derivative order d
    vector<Rect> obstacles;
    vector<vector<long long>> binom;       // binom[n][i]
    vector<uint64_t> base_pow;             // base_pow[i] = baseP^i
    vector<Action> actions;

    uint64_t baseP = 0;
    uint64_t action_count = 0;
    uint64_t mask_words = 0;
};

struct PrevFilter {
    int history_len = 0;
    const std::unordered_set<uint64_t>* winning = nullptr;
};

struct PreEdge {
    uint32_t pred = 0;
    uint32_t action_id = 0;
};

struct StageResult {
    int order_max = 0;
    int history_len = 0;
    size_t candidate_state_count = 0;
    size_t winning_state_count = 0;
    size_t no_locally_safe_action_count = 0;
    size_t propagated_loss_count = 0;
    size_t prev_filter_rejection_count = 0;
    std::unordered_set<uint64_t> winning_set;
    vector<uint64_t> winning_ids;
    vector<uint64_t> winning_masks_flat;   // winning_ids.size() * mask_words
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

struct LargeStageResult {
    int order_max = 0;
    int history_len = 0;
    size_t candidate_state_count = 0;
    size_t winning_state_count = 0;
    size_t no_locally_safe_action_count = 0;
    size_t propagated_loss_count = 0;
    vector<LargeState> winning_states;
    vector<uint64_t> winning_masks_flat;   // winning_states.size() * mask_words
};

struct CostSnapshot {
    std::chrono::steady_clock::time_point time;
    uint64_t current_rss_bytes = 0;
    uint64_t peak_rss_bytes = 0;
};

static constexpr uint64_t kUnknownMemory = std::numeric_limits<uint64_t>::max();

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
        die("unknown optional synthesis input section: " + tokens[0] + " ; expected obstacles");
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

static uint64_t currentResidentMemoryBytes() {
#if defined(__APPLE__) && defined(__MACH__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(),
                                 MACH_TASK_BASIC_INFO,
                                 reinterpret_cast<task_info_t>(&info),
                                 &count);
    if (kr != KERN_SUCCESS) return kUnknownMemory;
    return static_cast<uint64_t>(info.resident_size);
#elif defined(__linux__)
    ifstream statm("/proc/self/statm");
    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
    statm >> total_pages >> resident_pages;
    if (!statm) return kUnknownMemory;
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return kUnknownMemory;
    return resident_pages * static_cast<uint64_t>(page_size);
#else
    return kUnknownMemory;
#endif
}

static uint64_t peakResidentMemoryBytes() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return kUnknownMemory;
#if defined(__APPLE__) && defined(__MACH__)
    return static_cast<uint64_t>(usage.ru_maxrss);
#else
    return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
}

static CostSnapshot captureCost() {
    CostSnapshot snapshot;
    snapshot.time = std::chrono::steady_clock::now();
    snapshot.current_rss_bytes = currentResidentMemoryBytes();
    snapshot.peak_rss_bytes = peakResidentMemoryBytes();
    return snapshot;
}

static string formatSeconds(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3) << seconds << " s";
    return out.str();
}

static string formatMemory(uint64_t bytes) {
    if (bytes == kUnknownMemory) return "unknown";
    std::ostringstream out;
    out << std::fixed << std::setprecision(2)
        << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MiB";
    return out.str();
}

static string formatMemoryDelta(uint64_t before, uint64_t after) {
    if (before == kUnknownMemory || after == kUnknownMemory) return "unknown";
    std::ostringstream out;
    double delta_mib = static_cast<double>(after) - static_cast<double>(before);
    delta_mib /= 1024.0 * 1024.0;
    if (delta_mib >= 0.0) out << '+';
    out << std::fixed << std::setprecision(2) << delta_mib << " MiB";
    return out.str();
}

static void printCost(const string& label, const CostSnapshot& start) {
    CostSnapshot end = captureCost();
    std::chrono::duration<double> elapsed = end.time - start.time;
    cerr << label << ": elapsed " << formatSeconds(elapsed.count())
         << ", RSS " << formatMemory(end.current_rss_bytes)
         << " (" << formatMemoryDelta(start.current_rss_bytes, end.current_rss_bytes) << ")"
         << ", peak RSS " << formatMemory(end.peak_rss_bytes)
         << " (" << formatMemoryDelta(start.peak_rss_bytes, end.peak_rss_bytes) << ")\n";
}

static uint64_t checkedMul(uint64_t a, uint64_t b, const string& what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        die("uint64 overflow while computing " + what);
    }
    return a * b;
}

static uint64_t checkedAdd(uint64_t a, uint64_t b, const string& what) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        die("uint64 overflow while computing " + what);
    }
    return a + b;
}

static bool basePowersFitInUint64(int x_max, int y_max, int history_len) {
    if (x_max < 0 || y_max < 0 || history_len < 0) return false;
    uint64_t baseP = 0;
    uint64_t x_count = static_cast<uint64_t>(x_max) + 1ULL;
    uint64_t y_count = static_cast<uint64_t>(y_max) + 1ULL;
    if (x_count != 0 && y_count > std::numeric_limits<uint64_t>::max() / x_count) {
        return false;
    }
    baseP = x_count * y_count;

    uint64_t value = 1ULL;
    for (int i = 1; i <= history_len; ++i) {
        if (baseP != 0 && value > std::numeric_limits<uint64_t>::max() / baseP) {
            return false;
        }
        value *= baseP;
    }
    return true;
}

static uint32_t solverCode(const string& mode) {
    if (mode == "basic") return 0;
    if (mode == "optimized") return 1;
    if (mode == "optimized-iterative") return 2;
    die("unknown solver_mode: " + mode + " ; expected basic, optimized, or optimized-iterative");
    return 0;
}

static void precompute(Params& p, int max_history_len, bool require_base_powers) {
    if (p.x_max < 0 || p.y_max < 0) die("x_max and y_max must be nonnegative");
    if (p.vx_cmd_max < 0 || p.vy_cmd_max < 0) die("velocity command maxima must be nonnegative");
    if (p.eta_max < 0) die("eta_max must be nonnegative");
    if (p.k < 0) die("k must be nonnegative");
    if (p.x_wall < 0) die("x_wall must be nonnegative");
    if (p.x_wall > p.x_max) {
        cerr << "warning: x_wall > x_max; using the state-space bound x_max as the effective x upper bound.\n";
    }

    p.baseP = checkedMul(static_cast<uint64_t>(p.x_max) + 1ULL,
                         static_cast<uint64_t>(p.y_max) + 1ULL,
                         "number of positions");

    p.base_pow.assign(1, 1ULL);
    if (require_base_powers) {
        p.base_pow.assign(static_cast<size_t>(max_history_len) + 1, 1ULL);
        for (int i = 1; i <= max_history_len; ++i) {
            p.base_pow[static_cast<size_t>(i)] = checkedMul(p.base_pow[static_cast<size_t>(i - 1)], p.baseP, "base powers");
        }
    }

    p.binom.assign(static_cast<size_t>(p.k) + 1, vector<long long>(static_cast<size_t>(p.k) + 1, 0));
    for (int n = 0; n <= p.k; ++n) {
        p.binom[static_cast<size_t>(n)][0] = 1;
        p.binom[static_cast<size_t>(n)][static_cast<size_t>(n)] = 1;
        for (int i = 1; i < n; ++i) {
            __int128 val = static_cast<__int128>(p.binom[static_cast<size_t>(n - 1)][static_cast<size_t>(i - 1)])
                         + static_cast<__int128>(p.binom[static_cast<size_t>(n - 1)][static_cast<size_t>(i)]);
            if (val > std::numeric_limits<long long>::max()) die("binomial coefficient overflow; k is too large");
            p.binom[static_cast<size_t>(n)][static_cast<size_t>(i)] = static_cast<long long>(val);
        }
    }

    p.action_count = checkedMul(static_cast<uint64_t>(p.vx_cmd_max) + 1ULL,
                                static_cast<uint64_t>(p.vy_cmd_max) + 1ULL,
                                "number of actions");
    if (p.action_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        die("too many actions for uint32_t action ids");
    }
    p.mask_words = (p.action_count + 63ULL) / 64ULL;

    p.actions.clear();
    p.actions.reserve(static_cast<size_t>(p.action_count));
    uint32_t aid = 0;
    for (int vx = 0; vx <= p.vx_cmd_max; ++vx) {
        for (int vy = 0; vy <= p.vy_cmd_max; ++vy) {
            Action a;
            a.vx = vx;
            a.vy = vy;
            a.id = aid++;
            a.eta_x = std::min(p.eta_max, vx);
            a.eta_y = std::min(p.eta_max, vy);
            p.actions.push_back(a);
        }
    }
}

static inline bool positionOK(const Params& p, const Pos& q) {
    if (q.x < 0 || q.x > p.x_max || q.y < 0 || q.y > p.y_max || q.x > p.x_wall) {
        return false;
    }
    for (const Rect& r : p.obstacles) {
        if (q.x >= r.x_min && q.x <= r.x_max &&
            q.y >= r.y_min && q.y <= r.y_max) {
            return false;
        }
    }
    return true;
}

static inline uint64_t posId(const Params& p, const Pos& q) {
    return static_cast<uint64_t>(q.x) * (static_cast<uint64_t>(p.y_max) + 1ULL) + static_cast<uint64_t>(q.y);
}

static inline Pos decodePos(const Params& p, uint64_t id) {
    uint64_t ybase = static_cast<uint64_t>(p.y_max) + 1ULL;
    Pos q;
    q.x = static_cast<int>(id / ybase);
    q.y = static_cast<int>(id % ybase);
    return q;
}

static void decodeHistory(const Params& p, uint64_t id, int history_len, vector<Pos>& hist) {
    hist.resize(static_cast<size_t>(history_len));
    for (int i = 0; i < history_len; ++i) {
        uint64_t pid = id % p.baseP;
        id /= p.baseP;
        hist[static_cast<size_t>(i)] = decodePos(p, pid);
    }
}

static inline uint64_t encodeNextHistory(const Params& p, uint64_t state_id, int history_len, const Pos& next) {
    // Current encoding is [p_t, p_{t-1}, ...].
    // Successor encoding is [p_{t+1}, p_t, ..., p_{t-history_len+2}].
    uint64_t prefix = 0;
    if (history_len > 1) {
        prefix = state_id % p.base_pow[static_cast<size_t>(history_len - 1)];
    }
    return checkedAdd(posId(p, next), checkedMul(p.baseP, prefix, "successor state id"), "successor state id");
}

static inline uint64_t projectId(const Params& p, uint64_t state_id, int target_history_len) {
    return state_id % p.base_pow[static_cast<size_t>(target_history_len)];
}

static bool transitionOK(const Params& p,
                         const vector<Pos>& hist,
                         const Pos& next,
                         int order_max) {
    if (!positionOK(p, next)) return false;
    if (order_max == 0) return true;
    if (static_cast<int>(hist.size()) < order_max) {
        die("internal error: history too short to check derivative order");
    }

    for (int d = 1; d <= order_max; ++d) {
        __int128 sx = next.x;
        __int128 sy = next.y;
        for (int i = 1; i <= d; ++i) {
            long long c = p.binom[static_cast<size_t>(d)][static_cast<size_t>(i)];
            if (i & 1) c = -c;
            sx += static_cast<__int128>(c) * hist[static_cast<size_t>(i - 1)].x;
            sy += static_cast<__int128>(c) * hist[static_cast<size_t>(i - 1)].y;
        }
        __int128 lhs = sx * sx + sy * sy;
        __int128 r = p.radius[static_cast<size_t>(d - 1)];
        __int128 rhs = r * r;
        if (lhs > rhs) return false;
    }
    return true;
}

static bool prevFilterOK(const Params& p,
                         const PrevFilter* prev,
                         uint64_t state_id) {
    if (prev == nullptr) return true;
    uint64_t pid = projectId(p, state_id, prev->history_len);
    return prev->winning->find(pid) != prev->winning->end();
}

static bool generateRobustLocalSuccessors(const Params& p,
                                          uint64_t state_id,
                                          int history_len,
                                          int order_max,
                                          const Action& a,
                                          const PrevFilter* prev,
                                          vector<Pos>& hist_buffer,
                                          vector<uint64_t>& succ_ids,
                                          size_t* prev_filter_rejections) {
    decodeHistory(p, state_id, history_len, hist_buffer);
    succ_ids.clear();

    const Pos cur = hist_buffer[0];
    for (int nx = -a.eta_x; nx <= a.eta_x; ++nx) {
        for (int ny = -a.eta_y; ny <= a.eta_y; ++ny) {
            Pos next;
            next.x = cur.x + a.vx + nx;
            next.y = cur.y + a.vy + ny;

            if (!transitionOK(p, hist_buffer, next, order_max)) {
                succ_ids.clear();
                return false;
            }

            uint64_t sid = encodeNextHistory(p, state_id, history_len, next);
            if (!prevFilterOK(p, prev, sid)) {
                if (prev_filter_rejections != nullptr) ++(*prev_filter_rejections);
                succ_ids.clear();
                return false;
            }
            succ_ids.push_back(sid);
        }
    }
    return !succ_ids.empty();
}

static LargeState nextLargeState(const LargeState& state, const Pos& next) {
    LargeState out;
    out.hist.resize(state.hist.size());
    out.hist[0] = next;
    for (size_t i = 1; i < state.hist.size(); ++i) {
        out.hist[i] = state.hist[i - 1];
    }
    return out;
}

static bool generateRobustLocalSuccessorsLarge(const Params& p,
                                               const LargeState& state,
                                               int order_max,
                                               const Action& a,
                                               vector<LargeState>& succ_states) {
    succ_states.clear();

    const Pos cur = state.hist[0];
    for (int nx = -a.eta_x; nx <= a.eta_x; ++nx) {
        for (int ny = -a.eta_y; ny <= a.eta_y; ++ny) {
            Pos next;
            next.x = cur.x + a.vx + nx;
            next.y = cur.y + a.vy + ny;

            if (!transitionOK(p, state.hist, next, order_max)) {
                succ_states.clear();
                return false;
            }

            succ_states.push_back(nextLargeState(state, next));
        }
    }
    return !succ_states.empty();
}

static inline bool maskGet(const vector<uint64_t>& masks_flat,
                           uint64_t mask_words,
                           uint32_t state_index,
                           uint32_t action_id) {
    size_t off = static_cast<size_t>(state_index) * static_cast<size_t>(mask_words) + static_cast<size_t>(action_id / 64U);
    return (masks_flat[off] >> (action_id & 63U)) & 1ULL;
}

static inline void maskSet(vector<uint64_t>& masks_flat,
                           uint64_t mask_words,
                           uint32_t state_index,
                           uint32_t action_id) {
    size_t off = static_cast<size_t>(state_index) * static_cast<size_t>(mask_words) + static_cast<size_t>(action_id / 64U);
    masks_flat[off] |= 1ULL << (action_id & 63U);
}

static inline void maskClear(vector<uint64_t>& masks_flat,
                             uint64_t mask_words,
                             uint32_t state_index,
                             uint32_t action_id) {
    size_t off = static_cast<size_t>(state_index) * static_cast<size_t>(mask_words) + static_cast<size_t>(action_id / 64U);
    masks_flat[off] &= ~(1ULL << (action_id & 63U));
}

static bool stageHasUsefulFilter(const StageResult& result) {
    return !(result.candidate_state_count > 0 &&
             result.candidate_state_count == result.winning_state_count &&
             result.no_locally_safe_action_count == 0 &&
             result.propagated_loss_count == 0);
}

static StageResult solveStage(const Params& p,
                              int order_max,
                              int history_len,
                              const PrevFilter* prev,
                              bool keep_masks) {
    cerr << "\n=== solving stage: derivative orders <= " << order_max
         << ", history length = " << history_len << " ===\n";

    StageResult result;
    result.order_max = order_max;
    result.history_len = history_len;

    std::unordered_map<uint64_t, uint32_t> index;
    index.reserve(1024);
    index.max_load_factor(0.70f);

    vector<uint64_t> states;
    states.reserve(1024);

    const uint64_t q0 = 0ULL;  // all positions are (0,0), so all position ids are zero
    if (!prevFilterOK(p, prev, q0)) {
        if (prev != nullptr) result.prev_filter_rejection_count = 1;
        cerr << "initial state is not in previous winning projection; stage is empty.\n";
        cerr << "previous-filter rejections: " << result.prev_filter_rejection_count << "\n";
        return result;
    }

    index.emplace(q0, 0U);
    states.push_back(q0);

    vector<Pos> hist;
    vector<uint64_t> succ_ids;
    size_t prev_filter_rejections = 0;

    for (size_t head = 0; head < states.size(); ++head) {
        uint64_t sid = states[head];
        for (const Action& a : p.actions) {
            bool robust = generateRobustLocalSuccessors(p, sid, history_len, order_max,
                                                        a, prev, hist, succ_ids,
                                                        &prev_filter_rejections);
            if (!robust) continue;
            for (uint64_t tid : succ_ids) {
                if (index.find(tid) == index.end()) {
                    if (states.size() >= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                        die("too many candidate states for uint32_t indices");
                    }
                    uint32_t idx = static_cast<uint32_t>(states.size());
                    index.emplace(tid, idx);
                    states.push_back(tid);
                }
            }
        }
    }

    result.candidate_state_count = states.size();
    result.prev_filter_rejection_count = prev_filter_rejections;
    cerr << "reachable locally safe candidate states: " << states.size() << "\n";
    cerr << "previous-filter rejections: " << prev_filter_rejections << "\n";

    const size_t n = states.size();
    vector<uint64_t> masks_flat;
    masks_flat.assign(n * static_cast<size_t>(p.mask_words), 0ULL);
    vector<uint32_t> live_action_count(n, 0U);
    vector<unsigned char> winning(n, 1U);
    vector<vector<PreEdge>> reverse(n);

    // Build initially locally-safe action graph.
    for (size_t i = 0; i < n; ++i) {
        uint64_t sid = states[i];
        for (const Action& a : p.actions) {
            bool robust = generateRobustLocalSuccessors(p, sid, history_len, order_max,
                                                        a, prev, hist, succ_ids,
                                                        nullptr);
            if (!robust) continue;

            bool all_known = true;
            for (uint64_t tid : succ_ids) {
                if (index.find(tid) == index.end()) {
                    all_known = false;
                    break;
                }
            }
            if (!all_known) continue;

            uint32_t si = static_cast<uint32_t>(i);
            maskSet(masks_flat, p.mask_words, si, a.id);
            ++live_action_count[i];

            for (uint64_t tid : succ_ids) {
                uint32_t tj = index.find(tid)->second;
                reverse[tj].push_back(PreEdge{si, a.id});
            }
        }
    }

    std::queue<uint32_t> losing_queue;
    size_t initially_losing = 0;
    for (size_t i = 0; i < n; ++i) {
        if (live_action_count[i] == 0U) {
            winning[i] = 0U;
            losing_queue.push(static_cast<uint32_t>(i));
            ++initially_losing;
        }
    }
    result.no_locally_safe_action_count = initially_losing;
    cerr << "states with no locally safe action: " << initially_losing << "\n";

    size_t propagated_losses = 0;
    while (!losing_queue.empty()) {
        uint32_t bad = losing_queue.front();
        losing_queue.pop();

        for (const PreEdge& e : reverse[bad]) {
            if (!winning[e.pred]) continue;
            if (!maskGet(masks_flat, p.mask_words, e.pred, e.action_id)) continue;

            maskClear(masks_flat, p.mask_words, e.pred, e.action_id);
            if (--live_action_count[e.pred] == 0U) {
                winning[e.pred] = 0U;
                losing_queue.push(e.pred);
                ++propagated_losses;
            }
        }
    }
    result.propagated_loss_count = propagated_losses;
    cerr << "additional states removed by fixed point: " << propagated_losses << "\n";

    size_t nwin = 0;
    for (size_t i = 0; i < n; ++i) {
        if (winning[i] && live_action_count[i] > 0U) ++nwin;
    }
    result.winning_state_count = nwin;
    cerr << "winning states after fixed point: " << nwin << "\n";

    result.winning_set.reserve(nwin * 2 + 1);
    result.winning_set.max_load_factor(0.70f);
    if (keep_masks) {
        result.winning_ids.reserve(nwin);
        result.winning_masks_flat.reserve(nwin * static_cast<size_t>(p.mask_words));
    }

    for (size_t i = 0; i < n; ++i) {
        if (!(winning[i] && live_action_count[i] > 0U)) continue;
        uint64_t sid = states[i];
        result.winning_set.insert(sid);
        if (keep_masks) {
            result.winning_ids.push_back(sid);
            size_t off = i * static_cast<size_t>(p.mask_words);
            for (uint64_t w = 0; w < p.mask_words; ++w) {
                result.winning_masks_flat.push_back(masks_flat[off + static_cast<size_t>(w)]);
            }
        }
    }

    if (result.winning_set.find(q0) == result.winning_set.end()) {
        cerr << "warning: initial state is losing at this stage.\n";
    } else {
        cerr << "initial state is winning at this stage.\n";
    }

    return result;
}

static LargeStageResult solveBasicLargeStage(const Params& p,
                                             int order_max,
                                             int history_len) {
    cerr << "\n=== solving stage: derivative orders <= " << order_max
         << ", history length = " << history_len
         << " (explicit large-state basic path) ===\n";

    LargeStageResult result;
    result.order_max = order_max;
    result.history_len = history_len;

    std::unordered_map<LargeState, uint32_t, LargeStateHash> index;
    index.reserve(1024);
    index.max_load_factor(0.70f);

    vector<LargeState> states;
    states.reserve(1024);

    LargeState q0;
    q0.hist.assign(static_cast<size_t>(history_len), Pos{0, 0});
    index.emplace(q0, 0U);
    states.push_back(q0);

    vector<LargeState> succ_states;

    for (size_t head = 0; head < states.size(); ++head) {
        LargeState state = states[head];
        for (const Action& a : p.actions) {
            bool robust = generateRobustLocalSuccessorsLarge(p, state, order_max, a, succ_states);
            if (!robust) continue;
            for (const LargeState& tid : succ_states) {
                if (index.find(tid) == index.end()) {
                    if (states.size() >= static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                        die("too many candidate states for uint32_t indices");
                    }
                    uint32_t idx = static_cast<uint32_t>(states.size());
                    index.emplace(tid, idx);
                    states.push_back(tid);
                }
            }
        }
    }

    result.candidate_state_count = states.size();
    cerr << "reachable locally safe candidate states: " << states.size() << "\n";
    cerr << "previous-filter rejections: 0\n";

    const size_t n = states.size();
    vector<uint64_t> masks_flat;
    masks_flat.assign(n * static_cast<size_t>(p.mask_words), 0ULL);
    vector<uint32_t> live_action_count(n, 0U);
    vector<unsigned char> winning(n, 1U);
    vector<vector<PreEdge>> reverse(n);

    // Build initially locally-safe action graph.
    for (size_t i = 0; i < n; ++i) {
        const LargeState& state = states[i];
        for (const Action& a : p.actions) {
            bool robust = generateRobustLocalSuccessorsLarge(p, state, order_max, a, succ_states);
            if (!robust) continue;

            bool all_known = true;
            for (const LargeState& tid : succ_states) {
                if (index.find(tid) == index.end()) {
                    all_known = false;
                    break;
                }
            }
            if (!all_known) continue;

            uint32_t si = static_cast<uint32_t>(i);
            maskSet(masks_flat, p.mask_words, si, a.id);
            ++live_action_count[i];

            for (const LargeState& tid : succ_states) {
                uint32_t tj = index.find(tid)->second;
                reverse[tj].push_back(PreEdge{si, a.id});
            }
        }
    }

    std::queue<uint32_t> losing_queue;
    size_t initially_losing = 0;
    for (size_t i = 0; i < n; ++i) {
        if (live_action_count[i] == 0U) {
            winning[i] = 0U;
            losing_queue.push(static_cast<uint32_t>(i));
            ++initially_losing;
        }
    }
    result.no_locally_safe_action_count = initially_losing;
    cerr << "states with no locally safe action: " << initially_losing << "\n";

    size_t propagated_losses = 0;
    while (!losing_queue.empty()) {
        uint32_t bad = losing_queue.front();
        losing_queue.pop();

        for (const PreEdge& e : reverse[bad]) {
            if (!winning[e.pred]) continue;
            if (!maskGet(masks_flat, p.mask_words, e.pred, e.action_id)) continue;

            maskClear(masks_flat, p.mask_words, e.pred, e.action_id);
            if (--live_action_count[e.pred] == 0U) {
                winning[e.pred] = 0U;
                losing_queue.push(e.pred);
                ++propagated_losses;
            }
        }
    }
    result.propagated_loss_count = propagated_losses;
    cerr << "additional states removed by fixed point: " << propagated_losses << "\n";

    size_t nwin = 0;
    for (size_t i = 0; i < n; ++i) {
        if (winning[i] && live_action_count[i] > 0U) ++nwin;
    }
    result.winning_state_count = nwin;
    cerr << "winning states after fixed point: " << nwin << "\n";

    result.winning_states.reserve(nwin);
    result.winning_masks_flat.reserve(nwin * static_cast<size_t>(p.mask_words));
    for (size_t i = 0; i < n; ++i) {
        if (!(winning[i] && live_action_count[i] > 0U)) continue;
        result.winning_states.push_back(states[i]);
        size_t off = i * static_cast<size_t>(p.mask_words);
        for (uint64_t w = 0; w < p.mask_words; ++w) {
            result.winning_masks_flat.push_back(masks_flat[off + static_cast<size_t>(w)]);
        }
    }

    if (winning[0] && live_action_count[0] > 0U) {
        cerr << "initial state is winning at this stage.\n";
    } else {
        cerr << "warning: initial state is losing at this stage.\n";
    }

    return result;
}

template <typename T>
static void writeBinary(ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) die("failed while writing strategy file");
}

static void writeStrategy(const Params& p,
                          int final_history_len,
                          const StageResult& result,
                          const string& output_path) {
    ofstream out(output_path, std::ios::binary);
    if (!out) die("could not open output strategy file: " + output_path);

    const char magic[8] = {'G','S','T','R','A','T','K','2'};
    out.write(magic, 8);
    if (!out) die("failed while writing strategy file magic");

    const uint32_t version = p.obstacles.empty() ? 2U : 3U;
    writeBinary(out, version);
    writeBinary(out, p.solver_code);
    writeBinary(out, static_cast<uint32_t>(p.k));
    writeBinary(out, static_cast<uint32_t>(final_history_len));

    writeBinary(out, static_cast<int64_t>(p.x_max));
    writeBinary(out, static_cast<int64_t>(p.y_max));
    writeBinary(out, static_cast<int64_t>(p.vx_cmd_max));
    writeBinary(out, static_cast<int64_t>(p.vy_cmd_max));
    writeBinary(out, static_cast<int64_t>(p.eta_max));
    writeBinary(out, static_cast<int64_t>(p.x_wall));

    writeBinary(out, p.baseP);
    writeBinary(out, p.action_count);
    writeBinary(out, p.mask_words);

    for (int d = 0; d < p.k; ++d) {
        writeBinary(out, static_cast<int64_t>(p.radius[static_cast<size_t>(d)]));
    }

    if (version >= 3U) {
        uint64_t obstacle_count = static_cast<uint64_t>(p.obstacles.size());
        writeBinary(out, obstacle_count);
        for (const Rect& r : p.obstacles) {
            writeBinary(out, static_cast<int64_t>(r.x_min));
            writeBinary(out, static_cast<int64_t>(r.y_min));
            writeBinary(out, static_cast<int64_t>(r.x_max));
            writeBinary(out, static_cast<int64_t>(r.y_max));
        }
    }

    uint64_t nwin = static_cast<uint64_t>(result.winning_ids.size());
    writeBinary(out, nwin);

    for (size_t i = 0; i < result.winning_ids.size(); ++i) {
        writeBinary(out, result.winning_ids[i]);
        size_t off = i * static_cast<size_t>(p.mask_words);
        for (uint64_t w = 0; w < p.mask_words; ++w) {
            writeBinary(out, result.winning_masks_flat[off + static_cast<size_t>(w)]);
        }
    }

    cerr << "wrote strategy to: " << output_path << "\n";
    cerr << "winning states written: " << nwin << "\n";
    cerr << "action count: " << p.action_count << ", mask words per state: " << p.mask_words << "\n";
}

static void writeLargeBasicStrategy(const Params& p,
                                    int final_history_len,
                                    const LargeStageResult& result,
                                    const string& output_path) {
    ofstream out(output_path, std::ios::binary);
    if (!out) die("could not open output strategy file: " + output_path);

    const char magic[8] = {'G','S','T','R','A','T','K','2'};
    out.write(magic, 8);
    if (!out) die("failed while writing strategy file magic");

    // Version 4 stores explicit histories instead of uint64 base-P state ids.
    const uint32_t version = 4U;
    writeBinary(out, version);
    writeBinary(out, p.solver_code);
    writeBinary(out, static_cast<uint32_t>(p.k));
    writeBinary(out, static_cast<uint32_t>(final_history_len));

    writeBinary(out, static_cast<int64_t>(p.x_max));
    writeBinary(out, static_cast<int64_t>(p.y_max));
    writeBinary(out, static_cast<int64_t>(p.vx_cmd_max));
    writeBinary(out, static_cast<int64_t>(p.vy_cmd_max));
    writeBinary(out, static_cast<int64_t>(p.eta_max));
    writeBinary(out, static_cast<int64_t>(p.x_wall));

    writeBinary(out, p.baseP);
    writeBinary(out, p.action_count);
    writeBinary(out, p.mask_words);

    for (int d = 0; d < p.k; ++d) {
        writeBinary(out, static_cast<int64_t>(p.radius[static_cast<size_t>(d)]));
    }

    uint64_t obstacle_count = static_cast<uint64_t>(p.obstacles.size());
    writeBinary(out, obstacle_count);
    for (const Rect& r : p.obstacles) {
        writeBinary(out, static_cast<int64_t>(r.x_min));
        writeBinary(out, static_cast<int64_t>(r.y_min));
        writeBinary(out, static_cast<int64_t>(r.x_max));
        writeBinary(out, static_cast<int64_t>(r.y_max));
    }

    uint64_t nwin = static_cast<uint64_t>(result.winning_states.size());
    writeBinary(out, nwin);

    for (size_t i = 0; i < result.winning_states.size(); ++i) {
        const LargeState& state = result.winning_states[i];
        if (static_cast<int>(state.hist.size()) != final_history_len) {
            die("internal error: large strategy state has wrong history length");
        }
        for (const Pos& q : state.hist) {
            writeBinary(out, static_cast<int64_t>(q.x));
            writeBinary(out, static_cast<int64_t>(q.y));
        }
        size_t off = i * static_cast<size_t>(p.mask_words);
        for (uint64_t w = 0; w < p.mask_words; ++w) {
            writeBinary(out, result.winning_masks_flat[off + static_cast<size_t>(w)]);
        }
    }

    cerr << "wrote large-state basic strategy to: " << output_path << "\n";
    cerr << "strategy version 4 stores explicit histories for large basic-state ids.\n";
    cerr << "winning states written: " << nwin << "\n";
    cerr << "action count: " << p.action_count << ", mask words per state: " << p.mask_words << "\n";
}

int main() {
    try {
        Params p;
        string output_path;

        cin >> p.k;
        cin >> p.solver_mode;
        cin >> p.x_max >> p.y_max;
        cin >> p.vx_cmd_max >> p.vy_cmd_max;
        cin >> p.eta_max;
        cin >> p.x_wall;

        if (!cin) die("failed to read required parameters");
        p.solver_code = solverCode(p.solver_mode);

        p.radius.resize(static_cast<size_t>(p.k));
        for (int i = 0; i < p.k; ++i) {
            cin >> p.radius[static_cast<size_t>(i)];
            if (p.radius[static_cast<size_t>(i)] < 0) die("derivative radii must be nonnegative");
        }
        cin >> output_path;
        if (!cin) die("failed to read radii or output strategy path");

        vector<string> optional_tokens;
        string token;
        while (cin >> token) {
            optional_tokens.push_back(token);
        }
        if (cin.bad()) die("failed while reading optional synthesis input");
        p.obstacles = parseObstacleTail(optional_tokens, p.x_max, p.y_max);

        int max_history_len = 1;
        if (p.solver_mode == "basic") {
            max_history_len = std::max(1, p.k + 1);
        } else {
            max_history_len = std::max(1, p.k);
        }
        bool use_large_basic_state_ids = (p.solver_mode == "basic" &&
                                          !basePowersFitInUint64(p.x_max, p.y_max, max_history_len));
        CostSnapshot total_cost = captureCost();

        precompute(p, max_history_len, !use_large_basic_state_ids);

        int final_history_len = 1;

        if (use_large_basic_state_ids) {
            final_history_len = std::max(1, p.k + 1);
            LargeStageResult final_result = solveBasicLargeStage(p, p.k, final_history_len);
            writeLargeBasicStrategy(p, final_history_len, final_result, output_path);
        } else if (p.solver_mode == "basic") {
            final_history_len = std::max(1, p.k + 1);
            StageResult final_result;
            final_result = solveStage(p, p.k, final_history_len, nullptr, true);
            writeStrategy(p, final_history_len, final_result, output_path);
        } else if (p.solver_mode == "optimized") {
            final_history_len = std::max(1, p.k);
            StageResult final_result;
            final_result = solveStage(p, p.k, final_history_len, nullptr, true);
            writeStrategy(p, final_history_len, final_result, output_path);
        } else if (p.solver_mode == "optimized-iterative") {
            StageResult final_result;
            if (p.k == 1) {
                CostSnapshot iteration_cost = captureCost();
                final_history_len = 1;
                final_result = solveStage(p, 1, final_history_len, nullptr, true);
                printCost("optimized-iterative final stage d=1, history length=1 cost",
                          iteration_cost);
            } else {
                StageResult prev;
                bool have_prev = false;
                for (int d = 1; d <= p.k; ++d) {
                    CostSnapshot iteration_cost = captureCost();
                    int L = std::max(1, d);
                    bool keep = (d == p.k);
                    PrevFilter filter;
                    PrevFilter* filter_ptr = nullptr;
                    if (have_prev) {
                        filter.history_len = prev.history_len;
                        filter.winning = &prev.winning_set;
                        filter_ptr = &filter;
                    }
                    StageResult cur = solveStage(p, d, L, filter_ptr, keep);
                    if (keep) {
                        final_history_len = L;
                        final_result = std::move(cur);
                    } else if (stageHasUsefulFilter(cur)) {
                        StageResult compact_prev;
                        compact_prev.order_max = cur.order_max;
                        compact_prev.history_len = cur.history_len;
                        compact_prev.candidate_state_count = cur.candidate_state_count;
                        compact_prev.winning_state_count = cur.winning_state_count;
                        compact_prev.no_locally_safe_action_count = cur.no_locally_safe_action_count;
                        compact_prev.propagated_loss_count = cur.propagated_loss_count;
                        compact_prev.prev_filter_rejection_count = cur.prev_filter_rejection_count;
                        compact_prev.winning_set = std::move(cur.winning_set);
                        prev = std::move(compact_prev);
                        have_prev = true;
                        cerr << "optimized-iterative filter retained from d=" << d
                             << " with " << prev.winning_state_count << " winning states.\n";
                    } else {
                        cerr << "optimized-iterative filter skipped for d=" << d
                             << " because this stage removed no states.\n";
                    }
                    printCost("optimized-iterative iteration d=" + std::to_string(d) +
                              ", history length=" + std::to_string(L) + " cost",
                              iteration_cost);
                }
            }
            writeStrategy(p, final_history_len, final_result, output_path);
        }

        printCost("synthesis total cost", total_cost);
        return 0;
    } catch (const std::exception& e) {
        cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
