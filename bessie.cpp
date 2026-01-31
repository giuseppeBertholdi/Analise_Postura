#include <bits/stdc++.h>
using namespace std;

int N, F;
vector<int> a;
vector<vector<int>> single_danger;
vector<vector<pair<int, int>>> periodic_danger;
vector<bool> on_farmer_cycle;

inline bool is_dangerous(int f, int t) {
    auto& sd = single_danger[f];
    if (!sd.empty() && binary_search(sd.begin(), sd.end(), t)) return true;
    for (auto& [first, period] : periodic_danger[f]) {
        if (t >= first && (t - first) % period == 0) return true;
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    cin >> N >> F;
    
    a.resize(N + 1);
    for (int i = 1; i <= N; i++) {
        cin >> a[i];
    }
    
    vector<int> farmer_start(F);
    vector<bool> farmer_at_start(N + 1, false);
    for (int i = 0; i < F; i++) {
        cin >> farmer_start[i];
        farmer_at_start[farmer_start[i]] = true;
    }
    
    single_danger.resize(N + 1);
    periodic_danger.resize(N + 1);
    on_farmer_cycle.resize(N + 1, false);
    
    // Process each farmer's path
    for (int s : farmer_start) {
        vector<int> path;
        vector<int> visit_time(N + 1, -1);
        int cur = s;
        int t = 0;
        
        while (visit_time[cur] == -1) {
            visit_time[cur] = t;
            path.push_back(cur);
            cur = a[cur];
            t++;
        }
        
        int cycle_start_time = visit_time[cur];
        int clen = t - cycle_start_time;
        
        for (int i = 0; i < (int)path.size(); i++) {
            if (i < cycle_start_time) {
                single_danger[path[i]].push_back(i);
            } else {
                on_farmer_cycle[path[i]] = true;
                periodic_danger[path[i]].push_back({i, clen});
            }
        }
    }
    
    for (int i = 1; i <= N; i++) {
        sort(single_danger[i].begin(), single_danger[i].end());
        single_danger[i].erase(unique(single_danger[i].begin(), single_danger[i].end()), single_danger[i].end());
    }
    
    // Precompute dist_to_safe using reverse BFS
    vector<int> dist_to_safe(N + 1, -1);
    
    for (int i = 1; i <= N; i++) {
        if (!on_farmer_cycle[i]) {
            dist_to_safe[i] = 0;
        }
    }
    
    vector<vector<int>> rev(N + 1);
    for (int i = 1; i <= N; i++) {
        rev[a[i]].push_back(i);
    }
    
    queue<int> init_q;
    for (int i = 1; i <= N; i++) {
        if (dist_to_safe[i] == 0) {
            init_q.push(i);
        }
    }
    
    while (!init_q.empty()) {
        int u = init_q.front();
        init_q.pop();
        for (int v : rev[u]) {
            if (dist_to_safe[v] == -1) {
                dist_to_safe[v] = dist_to_safe[u] + 1;
                init_q.push(v);
            }
        }
    }
    
    // Process each starting farm
    for (int b = 1; b <= N; b++) {
        if (farmer_at_start[b]) {
            cout << -1 << "\n";
            continue;
        }
        
        bool has_safe_farm = (dist_to_safe[b] != -1);
        
        if (has_safe_farm) {
            // Quick check: can reach safe farm directly?
            int safe_dist = dist_to_safe[b];
            bool can_reach_directly = true;
            int cur = b;
            for (int t = 0; t <= safe_dist; t++) {
                if (is_dangerous(cur, t)) {
                    can_reach_directly = false;
                    break;
                }
                if (t < safe_dist) cur = a[cur];
            }
            
            if (can_reach_directly) {
                cout << -2 << "\n";
                continue;
            }
        }
        
        // Need BFS
        vector<int> bessie_path;
        vector<int> visit_idx(N + 1, -1);
        int cur = b;
        while (visit_idx[cur] == -1) {
            visit_idx[cur] = bessie_path.size();
            bessie_path.push_back(cur);
            cur = a[cur];
        }
        int cycle_start_idx = visit_idx[cur];
        int bessie_cycle_len = (int)bessie_path.size() - cycle_start_idx;
        
        auto get_pos = [&](int m) -> int {
            if (m < (int)bessie_path.size()) return bessie_path[m];
            return bessie_path[cycle_start_idx + (m - cycle_start_idx) % bessie_cycle_len];
        };
        
        int max_moves = (int)bessie_path.size() + bessie_cycle_len;
        int max_rests_bound;
        
        if (has_safe_farm) {
            int max_danger_time = 0;
            for (int f : bessie_path) {
                if (!single_danger[f].empty()) {
                    max_danger_time = max(max_danger_time, single_danger[f].back());
                }
            }
            max_rests_bound = max_danger_time + (int)bessie_path.size() + 100;
        } else {
            // All farms on path are on farmer cycles
            // Max rests is bounded by cycle length (once on cycle, resting closes the gap with farmers)
            max_rests_bound = bessie_cycle_len + (int)bessie_path.size();
        }
        
        // BFS with efficient storage
        vector<vector<bool>> visited;
        map<pair<int,int>, bool> visited_map;
        bool use_vector = (max_moves <= 3000 && max_rests_bound <= 3000);
        
        if (use_vector) {
            visited.assign(max_moves + 1, vector<bool>(max_rests_bound + 1, false));
        }
        
        auto mark = [&](int m, int r) -> bool {
            if (use_vector) {
                if (visited[m][r]) return false;
                visited[m][r] = true;
                return true;
            } else {
                auto key = make_pair(m, r);
                if (visited_map.count(key)) return false;
                visited_map[key] = true;
                return true;
            }
        };
        
        queue<pair<int, int>> q;
        
        if (!is_dangerous(b, 0) && mark(0, 0)) {
            q.push({0, 0});
        }
        
        int max_rests = -1;
        bool found_infinite = false;
        
        while (!q.empty() && !found_infinite) {
            auto [m, r] = q.front();
            q.pop();
            
            int t = m + r;
            max_rests = max(max_rests, r);
            
            int cur_farm = get_pos(m);
            
            if (has_safe_farm && !on_farmer_cycle[cur_farm]) {
                int last_danger = single_danger[cur_farm].empty() ? -1 : single_danger[cur_farm].back();
                if (t > last_danger) {
                    found_infinite = true;
                    break;
                }
            }
            
            // Try resting
            if (r + 1 <= max_rests_bound && !is_dangerous(cur_farm, t + 1)) {
                if (mark(m, r + 1)) {
                    q.push({m, r + 1});
                }
            }
            
            // Try moving
            if (m + 1 <= max_moves) {
                int next_farm = get_pos(m + 1);
                if (!is_dangerous(next_farm, t + 1)) {
                    if (mark(m + 1, r)) {
                        q.push({m + 1, r});
                    }
                }
            }
        }
        
        if (found_infinite) {
            cout << -2 << "\n";
        } else {
            cout << max_rests << "\n";
        }
    }
    
    return 0;
}
