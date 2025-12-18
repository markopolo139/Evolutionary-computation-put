#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <string_view>
#include <optional>
#include <sstream>
#include <numeric>
#include <cmath>
#include <map>
#include <functional>
#include <chrono>
#include <tuple>
#include <set>
#include <iomanip>

struct Point {
    int x, y, cost;
};

using Points = std::vector<Point>;
using DistanceMatrix = std::vector<std::vector<int>>; 
using Solution = std::vector<int>;
using Solutions = std::vector<Solution>;
using CandidateMatrix = std::vector<std::vector<int>>; 

inline auto read_points(auto& input) {
    Points points;

    if (!input.is_open()) {
        std::cerr << "Error: Could not open input file" << std::endl;
        return points;
    }

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string segment;
        Point point;
        int field_count = 0;

        while (std::getline(ss, segment, ';')) {
            try {
                int value = std::stoi(segment);
                
                if (field_count == 0) {
                    point.x = value;
                } else if (field_count == 1) {
                    point.y = value;
                } else if (field_count == 2) {
                    point.cost = value;
                }
                field_count++;
                
            } catch (const std::exception& e) {
                std::cerr << "Skipping invalid entry on line: " << line << std::endl;
                field_count = -1;
                break;
            }
        }

        if (field_count == 3) {
            points.push_back(point);
        } else if (field_count > 0) {
            std::cerr << "Skipping line with incorrect number of fields: " << line << std::endl;
        }
    }

    return points;
}

inline auto calculate_distance_matrix(const Points& points) {
    DistanceMatrix distance_mat;
    distance_mat.reserve(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        distance_mat.emplace_back();
        distance_mat.back().reserve(points.size());
        for (size_t j = 0; j < points.size(); ++j) {
            if (i == j) {
                distance_mat[i].emplace_back(std::numeric_limits<int>::max());
                continue;
            }

            auto A = points[i];
            auto B = points[j];

            int dx = A.x - B.x;
            int dy = A.y - B.y;

            int dist = round(sqrt(dx*dx + dy*dy));
            distance_mat[i].emplace_back(dist);
        }
    }

    return distance_mat;
}

inline auto calculate_candidate_matrix(const DistanceMatrix& distance_mat, int k) {
    CandidateMatrix candidates(distance_mat.size());
    for (size_t i = 0; i < distance_mat.size(); ++i) {
        std::vector<std::pair<int, int>> sorted_nodes;
        sorted_nodes.reserve(distance_mat.size());
        for (size_t j = 0; j < distance_mat.size(); ++j) {
            if (i != j) {
                sorted_nodes.push_back({distance_mat[i][j], (int)j});
            }
        }
        std::sort(sorted_nodes.begin(), sorted_nodes.end());
        
        candidates[i].reserve(k);
        for (size_t j = 0; j < std::min((size_t)k, sorted_nodes.size()); ++j) {
            candidates[i].push_back(sorted_nodes[j].second);
        }
    }
    return candidates;
}

#ifndef DONT_PRINT_LATEX
std::map<std::string, Solution> best_solutions;
#endif

inline void export_solution(const Points& points, const Solution& solution, std::string method, const std::string& instance) {
#ifdef DONT_PRINT_LATEX
    std::cout << "Best:  ";
    for (int i : solution)
        std::cout << i << ", ";
    std::cout << solution.front() << std::endl;
#else
    best_solutions[method] = solution;
#endif

    std::ofstream points_csv(instance + "_points_" + method + ".csv");
    points_csv << "index,x,y,cost,selected" << std::endl;
    for (size_t i = 0; i < points.size(); ++i) {
        bool selected = std::find(solution.begin(), solution.end(), i) != solution.end();
        const Point& point = points[i];
        points_csv << i << "," << point.x << "," << point.y << "," << point.cost << "," << selected << std::endl;
    }

    std::ofstream solution_csv(instance + "_solution_" + method + ".csv");
    for (size_t i = 0; i < solution.size(); ++i) {
        const Point& from = points[solution[i]];
        const Point& to = points[solution[(i+1) % solution.size()]] ;

        solution_csv << from.x << "," << from.y << "\n";
        solution_csv << to.x   << "," << to.y   << "\n";
        solution_csv << std::endl;
    }
}

inline auto calculate_objective_function(const DistanceMatrix& distance_mat, const Solution& solution, const std::vector<int>& node_costs) {
    int result = 0;
    for (size_t i = 0; i < solution.size() - 1; ++i) {
        result += distance_mat[solution[i]][solution[i+1]] + node_costs[solution[i+1]];
    }
    result += distance_mat[solution.back()][solution.front()] + node_costs[solution.front()];
    return result;
}

inline void calculate_statistics(const Points& points, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, const Solutions& solutions, const std::string& instance, std::string_view method_short, std::string_view method) {
    int min = std::numeric_limits<int>::max();
    int max = std::numeric_limits<int>::min();
    size_t sum = 0;
    std::optional<Solution> best{ std::nullopt };

    for (const Solution& solution : solutions) {
        int objective_function = calculate_objective_function(distance_mat, solution, node_costs);

        if (objective_function <= min) {
            min = objective_function;
            best = solution;
        }

        if (objective_function >= max)
            max = objective_function;

        sum += objective_function;
    }

    double avg = sum / (double)solutions.size();

#ifdef DONT_PRINT_LATEX
    std::cout << "Method: " << method << std::endl;
    std::cout << "Scores: " << avg << " (" << min << " - " << max << ")" << std::endl;
#else
    std::cout << method << " & " << min << " & " << avg << " & " << max << " \\";
#endif

    if (best.has_value())
        export_solution(points, *best, std::string(method_short), instance);
}

inline void calculate_and_print_time_statistics(const std::vector<std::chrono::duration<double, std::milli>>& times) {
    if (times.empty()) {
        return;
    }

    double min_time = std::numeric_limits<double>::max();
    double max_time = std::numeric_limits<double>::min();
    double sum_time = 0;

    for (const auto& time : times) {
        double time_ms = time.count();
        if (time_ms < min_time) {
            min_time = time_ms;
        }
        if (time_ms > max_time) {
            max_time = time_ms;
        }
        sum_time += time_ms;
    }

    double avg_time = sum_time / times.size();
    std::cout << "Time: " << avg_time << "ms (" << min_time << "ms - " << max_time << "ms)" << std::endl;
}

class ProgressBar {
public:
    ProgressBar(int progress, int total, int bar_width = 50)
        : progress(progress), total(total), bar_width(bar_width) {}

    friend std::ostream& operator<<(std::ostream& os, const ProgressBar& bar) {
        float percentage = 0.0f;
        if (bar.total > 0) {
            percentage = static_cast<float>(bar.progress) / bar.total;
        }
        
        int barProgress = static_cast<int>(bar.bar_width * percentage);

        os << "[";
        for (int i = 0; i < barProgress; ++i)
            os << "=";
        os << ">";
        for (int i = 0; i < bar.bar_width - barProgress - 1; ++i)
            os << " ";
        os << "]";

        return os;
    }

private:
    int progress;
    int total;
    int bar_width;
};

inline void print_ls_runs_staticits(const std::vector<size_t>& ls_runs) {
    if (ls_runs.empty()) {
        return;
    }

    size_t min_ls_runs = std::numeric_limits<size_t>::max();
    size_t max_ls_runs = std::numeric_limits<size_t>::min();
    size_t sum_ls_runs = 0;

    for (const auto& i_ls_runs : ls_runs) {
        if (i_ls_runs < min_ls_runs) {
            min_ls_runs = i_ls_runs;
        }
        if (i_ls_runs > max_ls_runs) {
            max_ls_runs = i_ls_runs;
        }
        sum_ls_runs += i_ls_runs;
    }

    size_t avg_ls_runs = sum_ls_runs / ls_runs.size();
    std::cout << "Number of LS runs: " << avg_ls_runs << " (" << min_ls_runs << " - " << max_ls_runs << ")" << std::endl;
}

// SOLUTIONS
inline auto local_search_steepest_edges(Solution solution, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs) {
    bool improvement = true;
    while (improvement) {
        improvement = false;
        int best_delta = 0;
        std::function<void()> best_move;

        // Intra-route moves (edge exchange)
        for (size_t i = 0; i < solution.size(); ++i) {
            for (size_t j = i + 1; j < solution.size(); ++j) {
                int u1 = solution[i];
                int v1 = solution[(i + 1) % solution.size()];
                int u2 = solution[j];
                int v2 = solution[(j + 1) % solution.size()];
                if (v1 == u2 || u1 == v2) continue;

                int delta = distance_mat[u1][u2] + distance_mat[v1][v2] -
                            (distance_mat[u1][v1] + distance_mat[u2][v2]);

                if (delta < best_delta) {
                    best_delta = delta;
                    best_move = [=, &solution]() {
                        std::reverse(solution.begin() + i + 1, solution.begin() + j + 1);
                    };
                }
            }
        }

        // Inter-route moves
        std::vector<int> non_solution_nodes;
        std::vector<bool> in_solution(distance_mat.size(), false);
        for (int node : solution) in_solution[node] = true;
        for (size_t i = 0; i < distance_mat.size(); ++i) {
            if (!in_solution[i]) non_solution_nodes.push_back(i);
        }

        for (size_t i = 0; i < solution.size(); ++i) {
            for (int non_sol_node : non_solution_nodes) {
                int prev = (i == 0) ? solution.back() : solution[i - 1];
                int next = (i == solution.size() - 1) ? solution.front() : solution[i + 1];
                
                int delta = (distance_mat[prev][non_sol_node] + distance_mat[non_sol_node][next] + node_costs[non_sol_node]) -
                            (distance_mat[prev][solution[i]] + distance_mat[solution[i]][next] + node_costs[solution[i]]);

                if (delta < best_delta) {
                    best_delta = delta;
                    best_move = [=, &solution]() {
                        solution[i] = non_sol_node;
                    };
                }
            }
        }

        if (best_move) {
            best_move();
            improvement = true;
        }
    }

    return solution;
}

// Optimized Local Search with Candidate Lists and DLB
// Returns tuple<Solution, int> where int is number of "moves" or just calls?
// User wants "Number of LS runs". 
// In MSLS/ILS context, an "LS run" is usually one descent to local optimum.
// So this function just performs the descent.
inline Solution local_search_fast(Solution sol, const DistanceMatrix& dist, const std::vector<int>& node_costs, const CandidateMatrix& candidates) {
    int n_sol = (int)sol.size();
    int n_total = (int)dist.size();
    
    // Position lookup
    std::vector<int> pos(n_total, -1);
    for(int i=0; i<n_sol; ++i) pos[sol[i]] = i;

    // Don't Look Bits
    std::vector<bool> dlb(n_total, false);
    
    bool improved = true;
    while(improved) {
        improved = false;

        for (int i = 0; i < n_sol; ++i) {
            int u = sol[i];
            if (dlb[u]) continue;

            bool move_found = false;

            // Neighbors in solution
            int idx_u = i;
            int idx_prev_u = (i - 1 + n_sol) % n_sol;
            int idx_next_u = (i + 1) % n_sol;
            int prev_u = sol[idx_prev_u];
            int next_u = sol[idx_next_u];

            for (int v : candidates[u]) {
                if (v == prev_u || v == next_u) continue; 

                int idx_v = pos[v];

                if (idx_v != -1) {
                    // v is in solution (Intra-route 2-opt)
                    int idx_next_v = (idx_v + 1) % n_sol;
                    int next_v = sol[idx_next_v];

                    if (next_v != u) { 
                        int delta = dist[u][v] + dist[next_u][next_v] - (dist[u][next_u] + dist[v][next_v]);
                        if (delta < 0) {
                            // Apply 2-opt: reverse segment between next_u and v
                            int t_start = idx_next_u;
                            int t_end = idx_v;
                            int num_swaps = ((t_end - t_start + n_sol) % n_sol + 1) / 2;
                            for(int k=0; k<num_swaps; ++k) {
                                int p1 = (t_start + k) % n_sol;
                                int p2 = (t_end - k + n_sol) % n_sol;
                                int val1 = sol[p1];
                                int val2 = sol[p2];
                                std::swap(sol[p1], sol[p2]);
                                pos[val1] = p2;
                                pos[val2] = p1;
                            }
                            
                            dlb[u] = false; dlb[v] = false; 
                            dlb[next_u] = false; dlb[next_v] = false;
                            move_found = true;
                            break; 
                        }
                    }
                } else {
                    // v is NOT in solution (Inter-route Swap)
                    int delta = (dist[prev_u][v] + dist[v][next_u] + node_costs[v]) -
                                (dist[prev_u][u] + dist[u][next_u] + node_costs[u]);
                    
                    if (delta < 0) {
                        sol[idx_u] = v;
                        pos[u] = -1;
                        pos[v] = idx_u;
                        
                        dlb[v] = false; 
                        dlb[prev_u] = false; 
                        dlb[next_u] = false;
                        move_found = true;
                        break;
                    }
                }
            }

            if (move_found) {
                improved = true;
            } else {
                dlb[u] = true;
            }
        }
    }
    return sol;
}

inline auto generate_random_solution(int solution_length, int points_length, auto& random_engine) {
    Solution solution(points_length);
    std::iota(solution.begin(), solution.end(), 0);
    std::shuffle(solution.begin(), solution.end(), random_engine);
    solution.resize(solution_length);

    return solution;
}

inline auto multiple_start_local_search(size_t max_iters, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, int solution_length, int points_length, auto& random_engine) {
    Solution best;
    int best_score{ std::numeric_limits<int>::max() };
    do {
        Solution current{
            local_search_steepest_edges(
                generate_random_solution(solution_length, points_length, random_engine),
                distance_mat,
                node_costs
            )
        };
        int current_score{ calculate_objective_function(distance_mat, current, node_costs) };
        if (current_score < best_score) {
            best_score = current_score;
            best = std::move(current);
        }
    } while (--max_iters != 0);
    return best;
}

// Robust Crossover (Edge Recombination Style)
inline Solution crossover_edge_recombination(const Solution& p1, const Solution& p2, int solution_length, int points_length, const DistanceMatrix& dist, auto& rng) {
    // Collect edges from both parents
    // We treat edges as undirected: (u, v) == (v, u)
    // Map: node -> list of neighbors in p1/p2
    std::vector<std::vector<int>> neighbors(points_length); // for all points, though only some are in p1/p2
    
    auto add_edge = [&](int u, int v) {
        // Only add if not duplicate (max 4 neighbors per node potentially, usually 2 to 4)
        bool exists = false;
        for(int n : neighbors[u]) if(n == v) exists = true;
        if(!exists) neighbors[u].push_back(v);
        
        exists = false;
        for(int n : neighbors[v]) if(n == u) exists = true;
        if(!exists) neighbors[v].push_back(u);
    };

    for(size_t i=0; i<p1.size(); ++i) add_edge(p1[i], p1[(i+1)%p1.size()]);
    for(size_t i=0; i<p2.size(); ++i) add_edge(p2[i], p2[(i+1)%p2.size()]);

    Solution child;
    child.reserve(solution_length);
    std::vector<bool> in_child(points_length, false);

    // Start with a random node from p1 (to bias towards p1's structure if p2 is diverse)
    // or just p1[0]
    int current = p1[0]; 
    child.push_back(current);
    in_child[current] = true;

    while(child.size() < (size_t)solution_length) {
        // Find neighbor of current with fewest edges in neighbor map (that is not in child)
        int best_next = -1;
        int min_neighbors = 10000;
        int best_dist = std::numeric_limits<int>::max();

        // Check explicit neighbors
        for(int neighbor : neighbors[current]) {
            if(!in_child[neighbor]) {
                int deg = 0;
                for(int n_of_n : neighbors[neighbor]) if(!in_child[n_of_n]) deg++;
                
                // Tie-breaking by distance (original Edge Recombination is just degree)
                // Adding distance heuristic helps convergence
                if (deg < min_neighbors) {
                    min_neighbors = deg;
                    best_next = neighbor;
                    best_dist = dist[current][neighbor];
                } else if (deg == min_neighbors) {
                    if (dist[current][neighbor] < best_dist) {
                        best_next = neighbor;
                        best_dist = dist[current][neighbor];
                    }
                }
            }
        }

        if (best_next == -1) {
            // Random unvisited node (from union of p1/p2 preferred?)
            // Let's try to pick from p1 or p2 unused nodes first
            std::vector<int> candidates;
            for(int x : p1) if(!in_child[x]) candidates.push_back(x);
            if(candidates.empty()) for(int x : p2) if(!in_child[x]) candidates.push_back(x);
            
            if(!candidates.empty()) {
                std::uniform_int_distribution<size_t> d(0, candidates.size()-1);
                best_next = candidates[d(rng)];
            } else { 
                 // Global fallback (shouldn't happen if p1/p2 have size K)
                 for(int i=0; i<points_length; ++i) if(!in_child[i]) { best_next = i; break; }
            }
        }
        
        current = best_next;
        child.push_back(current);
        in_child[current] = true;
    }
    
    return child;
}

inline auto run_hea_optimized(
    std::chrono::nanoseconds stop_duration,
    const DistanceMatrix& distance_mat,
    const std::vector<int>& node_costs,
    const CandidateMatrix& candidates,
    int solution_length,
    int points_length,
    auto& random_engine
) {
    auto start_time = std::chrono::steady_clock::now();
    size_t ls_runs = 0;
    
    const size_t POP_SIZE = 20; 
    std::vector<Solution> population;
    std::vector<int> fitness;
    
    // Initialize Population
    while(population.size() < POP_SIZE) {
        Solution sol = generate_random_solution(solution_length, points_length, random_engine);
        sol = local_search_fast(sol, distance_mat, node_costs, candidates); ls_runs++;
        int f = calculate_objective_function(distance_mat, sol, node_costs);
        
        bool unique = true;
        for(int existing_f : fitness) if(existing_f == f) unique = false;
        
        if(unique) {
            population.push_back(sol);
            fitness.push_back(f);
        }
    }
    
    // Sort population by fitness (best first)
    // We maintain this order or just find best/worst when needed.
    // Let's keep it simple: find indices.
    
    auto get_best_idx = [&]() {
        int min_f = std::numeric_limits<int>::max();
        int idx = -1;
        for(size_t i=0; i<fitness.size(); ++i) if(fitness[i] < min_f) { min_f = fitness[i]; idx = i; }
        return idx;
    };
    
    int best_idx = get_best_idx();
    Solution global_best = population[best_idx];
    int global_best_score = fitness[best_idx];
    
    int iterations_without_improvement = 0;
    
    while(std::chrono::steady_clock::now() - start_time < stop_duration) {
        // Selection: Binary Tournament for P1, Random for P2 (or Tournament P2)
        std::uniform_int_distribution<size_t> dist_idx(0, POP_SIZE - 1);
        int i1 = dist_idx(random_engine);
        int i2 = dist_idx(random_engine);
        int p1_idx = (fitness[i1] < fitness[i2]) ? i1 : i2;
        
        int p2_idx = dist_idx(random_engine);
        while(p2_idx == p1_idx) p2_idx = dist_idx(random_engine);
        
        // Crossover
        Solution child = crossover_edge_recombination(population[p1_idx], population[p2_idx], solution_length, points_length, distance_mat, random_engine);
        
        // Mutation: Heavy Kick if stagnating? Or always small kick?
        // Edge Recombination is greedy, so child might be locally optimal already or close.
        // Apply LS immediately? 
        // Standard Memetic Algorithm: Crossover -> Mutation -> LS
        
        if (std::uniform_real_distribution<>(0, 1)(random_engine) < 0.05) {
            // Double Bridge
             if (child.size() > 8) {
                std::uniform_int_distribution<int> cut_dist(1, child.size() - 1);
                std::vector<int> cuts;
                while(cuts.size() < 3) {
                    int c = cut_dist(random_engine);
                    bool ok = true; 
                    for(int existing : cuts) if(abs(existing - c) < 1) ok = false;
                    if(ok) cuts.push_back(c);
                }
                std::sort(cuts.begin(), cuts.end());
                Solution m; m.reserve(child.size());
                m.insert(m.end(), child.begin(), child.begin() + cuts[0]);
                m.insert(m.end(), child.begin() + cuts[2], child.end());
                m.insert(m.end(), child.begin() + cuts[1], child.begin() + cuts[2]);
                m.insert(m.end(), child.begin() + cuts[0], child.begin() + cuts[1]);
                child = m;
             }
        }
        
        // LS
        child = local_search_fast(child, distance_mat, node_costs, candidates); ls_runs++;
        int child_f = calculate_objective_function(distance_mat, child, node_costs);
        
        // Replacement Strategy: Steady State / Replace Worst
        // But maintain diversity (no duplicate fitness)
        bool exists = false;
        for(int f : fitness) if(f == child_f) exists = true;
        
        if (!exists) {
            // Replace worst if child is better than worst
            int max_f = -1;
            int worst_idx = -1;
            for(size_t i=0; i<fitness.size(); ++i) {
                if(fitness[i] > max_f) { max_f = fitness[i]; worst_idx = i; }
            }
            
            if (child_f < max_f) {
                population[worst_idx] = child;
                fitness[worst_idx] = child_f;
                
                if (child_f < global_best_score) {
                    global_best_score = child_f;
                    global_best = child;
                    iterations_without_improvement = 0;
                }
            }
        } else {
             iterations_without_improvement++;
        }
        
        // Restart / Cataclysm
        if (iterations_without_improvement > 300) {
            // Keep best, kill rest
             for(size_t i=0; i<POP_SIZE; ++i) {
                 if (fitness[i] == global_best_score) continue;
                 
                 Solution s = generate_random_solution(solution_length, points_length, random_engine);
                 s = local_search_fast(s, distance_mat, node_costs, candidates); ls_runs++;
                 population[i] = s;
                 fitness[i] = calculate_objective_function(distance_mat, s, node_costs);
             }
             iterations_without_improvement = 0;
        }
    }
    
    return std::make_tuple(global_best, ls_runs);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Error: This program requires exactly one argument. It should specify the name of the instance" << std::endl;
        return 1; 
    }
    std::string instance = argv[1];
    std::string instance_file = instance + ".csv";
    std::ifstream file(instance_file);
    if (!file.is_open()) return 1;

    auto points = read_points(file);
    if (points.empty()) return 1;
    auto distance_mat = calculate_distance_matrix(points);
    std::vector<int> node_costs;
    for(const auto& p : points) node_costs.push_back(p.cost);

    // Precompute Candidates
    auto candidates = calculate_candidate_matrix(distance_mat, 40);

    auto rng = std::mt19937{156053 + 156042};

    int solution_length = points.size() / 2; 
    
    Solutions solutions;
    std::vector<std::chrono::duration<double, std::milli>> times;
    std::vector<size_t> ls_runs;

    constexpr size_t max_runs = 20;

    // 1. MSLS (Reference for Time)
    std::cout << "Running MSLS" << std::flush;
    for (size_t i = 0; i < max_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        solutions.emplace_back(multiple_start_local_search(200, distance_mat, node_costs, solution_length, points.size(), rng));
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(end - start);
        std::cout << "\rRunning MSLS " << ProgressBar(i, max_runs, max_runs) << " " << std::chrono::duration_cast<std::chrono::seconds>(times.back()).count() * (max_runs - i - 1) << "s left     " << std::flush;
    }
    std::cout << std::endl;
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "msls", "msls");
    calculate_and_print_time_statistics(times);
    
    std::chrono::duration<double, std::milli> sum = std::accumulate(times.begin(), times.end(), std::chrono::duration<double, std::milli>(0.0));
    std::chrono::duration<double, std::milli> avg_ms = sum / times.size();
    auto avg_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(avg_ms);
    solutions.clear(); times.clear(); std::cout << std::endl;

    // 2. HEA Optimized (New)
    std::cout << "Running HEA Optimized" << std::flush;
    for (size_t i = 0; i < max_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        const auto result{ run_hea_optimized(avg_ns, distance_mat, node_costs, candidates, solution_length, points.size(), rng) };
        solutions.emplace_back(std::get<0>(result));
        ls_runs.emplace_back(std::get<1>(result));
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(end - start);
        std::cout << "\rRunning HEA Optimized " << ProgressBar(i, max_runs, max_runs) << " " << std::chrono::duration_cast<std::chrono::seconds>(times.back()).count() * (max_runs - i - 1) << "s left     " << std::flush;
    }
    std::cout << std::endl;
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "hea_opt", "hea_opt");
    calculate_and_print_time_statistics(times);
    print_ls_runs_staticits(ls_runs);
    solutions.clear(); ls_runs.clear(); times.clear(); std::cout << std::endl;

#ifndef DONT_PRINT_LATEX
    for (const auto& kv : best_solutions) {
        std::cout << std::get<0>(kv) << " & ";
        for (auto i : std::get<1>(kv))
            std::cout << i << ", ";
        std::cout << "\\" << std::endl;
    }
#endif

    return 0;
}
