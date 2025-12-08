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

struct Point {
    int x, y, cost;
};

using Points = std::vector<Point>;
using DistanceMatrix = std::vector<std::vector<int>>;
using Solution = std::vector<int>;
using Solutions = std::vector<Solution>;

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
    std::cout << method << " & " << min << " & " << avg << " & " << max << " \\\\";
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

inline auto perturb_solution(const Solution& solution, int points_length, int perturbation_strength, auto& random_engine) {
    auto perturbed_solution = solution;

    // Find nodes *not* in the solution
    std::vector<int> non_solution_nodes;
    std::vector<bool> in_solution(points_length, false);
    for (int node : solution) {
        in_solution[node] = true;
    }
    for (int i = 0; i < points_length; ++i) {
        if (!in_solution[i]) {
            non_solution_nodes.push_back(i);
        }
    }

    // If there are no nodes outside the solution, we can't perturb.
    if (non_solution_nodes.empty()) {
        return perturbed_solution;
    }

    // Ensure perturbation strength is not larger than what's possible
    int max_swaps = std::min({
        static_cast<int>(perturbed_solution.size()), 
        static_cast<int>(non_solution_nodes.size()), 
        perturbation_strength
    });

    std::uniform_int_distribution<size_t> sol_dist(0, perturbed_solution.size() - 1);
    std::uniform_int_distribution<size_t> non_sol_dist(0, non_solution_nodes.size() - 1);

    for (int i = 0; i < max_swaps; ++i) {
        // Select random indices to swap
        size_t idx_to_remove = sol_dist(random_engine);
        size_t idx_to_add = non_sol_dist(random_engine);

        // Get the actual node values
        int node_to_remove = perturbed_solution[idx_to_remove];
        int node_to_add = non_solution_nodes[idx_to_add];

        // Perform the swap
        perturbed_solution[idx_to_remove] = node_to_add;
        non_solution_nodes[idx_to_add] = node_to_remove; 
    }

    return perturbed_solution;
}

inline auto iterated_local_search(
    std::chrono::nanoseconds stop_duration,
    const DistanceMatrix& distance_mat,
    const std::vector<int>& node_costs,
    int solution_length,
    int points_length,
    auto& random_engine,
    int perturbation_strength = 21
) {
    auto start_time = std::chrono::steady_clock::now();
    size_t ls_runs{ 0 };

    // Generate an initial solution x
    Solution x = generate_random_solution(solution_length, points_length, random_engine);
    // x := Local search (x)
    x = local_search_steepest_edges(x, distance_mat, node_costs); ++ls_runs;

    int x_score = calculate_objective_function(distance_mat, x, node_costs);

    // Keep track of the best solution found overall
    Solution best_solution = x;
    int best_score = x_score;

    // Repeat until stopping conditions are met
    while (std::chrono::steady_clock::now() - start_time < stop_duration) {
        // y := Perturb (x)
        Solution y = perturb_solution(x, points_length, perturbation_strength, random_engine);
        
        // y := Local search (y)
        y = local_search_steepest_edges(std::move(y), distance_mat, node_costs); ++ls_runs;
        
        int y_score = calculate_objective_function(distance_mat, y, node_costs);

        // If f(y) < f(x) then (minimization problem)
        if (y_score < x_score) {
            x = std::move(y);
            x_score = y_score;

            // Update the overall best solution
            if (x_score < best_score) {
                best_score = x_score;
                best_solution = x; // This is a copy
            }
        }
    }

    return std::make_tuple(best_solution, ls_runs);
}

inline auto destroy(
    Solution solution,
    const DistanceMatrix& distance_mat,
    const std::vector<int>& node_costs,
    auto& rng,
    double strength,
    double cost_weight = 1.0, 
    double dist_weight = 1.0
) {
    int N = solution.size();
    if (N < 3 || strength <= 0.0) return solution;

    int total_to_remove = static_cast<int>(N * strength);
    // Safety: ensure we leave at least 2 nodes
    if (N - total_to_remove < 2) total_to_remove = N - 2;

    std::vector<bool> masked(N, false); // True = will be removed
    int currently_marked = 0;

    // --- 1. SELECTION PHASE (Marking) ---
    while (currently_marked < total_to_remove) {
        
        // 1a. Pick a random length for this specific segment
        // Min length 2, Max length = remaining quota (clamped)
        int remaining = total_to_remove - currently_marked;
        int max_len = std::min(remaining, N / 2); // Don't allow one chunk to eat half the path
        if (max_len < 1) break;

        std::uniform_int_distribution<int> len_dist(1, max_len);
        int window_len = len_dist(rng);

        // 1b. Calculate weights for all valid windows
        std::vector<double> weights;
        std::vector<int> start_indices;
        
        // We iterate through all possible start positions
        for (int i = 0; i < N; ++i) {
            
            // Check if this window overlaps with ALREADY masked nodes
            bool overlaps = false;
            for (int k = 0; k < window_len; ++k) {
                int idx = (i + k) % N; // Handle Wrap-Around
                if (masked[idx]) { 
                    overlaps = true; 

                    if (idx >= i) { // The loop's ++i will move us to idx + 1
                        i = idx; 
                    }

                    break; 
                }
            }
            
            if (overlaps) {
                // Cannot pick this window, it hits an existing hole
                continue; 
            }

            // Calculate Heuristic Score for this window
            double w_cost = 0.0;
            double w_dist = 0.0;

            for (int k = 0; k < window_len; ++k) {
                int curr = solution[(i + k) % N];
                // Next node in original path
                int next = solution[(i + k + 1) % N]; 
                w_cost += node_costs[curr];
                w_dist += distance_mat[curr][next];
            }

            // Weight formula: (Cost + Dist)^2 to favor expensive spots
            double score = (cost_weight * w_cost) + (dist_weight * w_dist);
            weights.push_back(score * score); 
            start_indices.push_back(i);
        }

        // If no valid windows found (path is Swiss cheese), stop
        if (weights.empty()) break;

        // 1c. Pick a window
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        int picked_idx = dist(rng);
        int start_node_idx = start_indices[picked_idx];

        // 1d. Mark it
        for (int k = 0; k < window_len; ++k) {
            int idx = (start_node_idx + k) % N;
            masked[idx] = true;
        }
        currently_marked += window_len;
    }

    // --- 2. RECONSTRUCTION PHASE (Removal) ---
    // Create new solution strictly preserving order of non-masked nodes
    std::vector<int> new_solution;
    new_solution.reserve(N - currently_marked);

    for (int i = 0; i < N; ++i) {
        if (!masked[i]) {
            new_solution.push_back(solution[i]);
        }
    }

    return new_solution;
}

inline auto nn_insert_weighted_regret(Solution solution, int solution_length, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, double regret_weight = 1.0, double cost_weight = 1.0) {
    solution.reserve(solution_length);

    std::vector<bool> visited(distance_mat.size(), false);
    for (int node : solution) {
        visited[node] = true;
    }

    while (solution.size() < solution_length) {
        int best_point_to_add = -1;
        size_t best_insertion_index = -1;
        double max_weighted_score = -std::numeric_limits<double>::max();

        for (size_t p = 0; p < distance_mat.size(); ++p) {
            if (!visited[p]) {
                int min_cost1 = std::numeric_limits<int>::max();
                int min_cost2 = std::numeric_limits<int>::max();
                size_t current_best_insertion_index = -1;

                for (size_t insert_index = 0; insert_index <= solution.size(); ++insert_index) {
                    int cost_increase;
                    if (insert_index == 0) {
                        cost_increase = distance_mat[p][solution.front()] + node_costs[p];
                    } else if (insert_index == solution.size()) {
                        cost_increase = distance_mat[solution.back()][p] + node_costs[p];
                    } else {
                        int from_node = solution[insert_index - 1];
                        int to_node = solution[insert_index];
                        cost_increase = distance_mat[from_node][p] + distance_mat[p][to_node] - distance_mat[from_node][to_node] + node_costs[p];
                    }

                    if (cost_increase < min_cost1) {
                        min_cost2 = min_cost1;
                        min_cost1 = cost_increase;
                        current_best_insertion_index = insert_index;
                    } else if (cost_increase < min_cost2) {
                        min_cost2 = cost_increase;
                    }
                }

                double regret = (min_cost2 == std::numeric_limits<int>::max()) ? min_cost1 : (min_cost2 - min_cost1);
                double weighted_score = regret_weight * regret - cost_weight * min_cost1;

                if (weighted_score > max_weighted_score) {
                    max_weighted_score = weighted_score;
                    best_point_to_add = p;
                    best_insertion_index = current_best_insertion_index;
                }
            }
        }

        if (best_point_to_add != -1) {
            solution.insert(solution.begin() + best_insertion_index, best_point_to_add);
            visited[best_point_to_add] = true;
        } else {
            break; // No unvisited nodes left to add
        }
    }

    return solution;
}

template<bool use_ls>
inline auto large_neighborhood_search(
    std::chrono::nanoseconds stop_duration,
    const DistanceMatrix& distance_mat,
    const std::vector<int>& node_costs,
    int solution_length,
    int points_length,
    auto& random_engine
) {
    auto start_time = std::chrono::steady_clock::now();
    size_t main_loop_runs{ 0 };

    // Generate an initial solution x
    Solution x = generate_random_solution(solution_length, points_length, random_engine);

    // x := Local search (x)
    x = local_search_steepest_edges(x, distance_mat, node_costs);

    int x_score = calculate_objective_function(distance_mat, x, node_costs);

    // Keep track of the best solution found overall
    Solution best_solution = x;
    int best_score = x_score;

    // Repeat until stopping conditions are met
    while (std::chrono::steady_clock::now() - start_time < stop_duration) {
        // y := Destroy(x)
        Solution y = destroy(x, distance_mat, node_costs, random_engine, 0.4);

        // y := Repair(y)
        y = nn_insert_weighted_regret(std::move(y), x.size(), distance_mat, node_costs);

        if constexpr (use_ls) {
            // y := Local search (y)
            y = local_search_steepest_edges(std::move(y), distance_mat, node_costs);
        }

        int y_score = calculate_objective_function(distance_mat, y, node_costs);

        // If f(y) < f(x) then (minimization problem)
        if (y_score < x_score) {
            x = std::move(y);
            x_score = y_score;

            // Update the overall best solution
            if (x_score < best_score) {
                best_score = x_score;
                best_solution = x; // This is a copy
            }
        }

        main_loop_runs++;
    }

    return std::make_tuple(best_solution, main_loop_runs);
}

// --- HEA Recombination Operators ---

// Operator 1: Common nodes and edges -> Random connection
inline Solution hea_operator_1(const Solution& p1, const Solution& p2, int solution_length, int points_length, auto& rng, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs) {
    // 1. Find common edges
    // Normalize edges for comparison (u, v) where u < v
    std::set<std::pair<int, int>> p2_edges;
    for (size_t i = 0; i < p2.size(); ++i) {
        int u = p2[i];
        int v = p2[(i + 1) % p2.size()];
        if (u > v) std::swap(u, v);
        p2_edges.insert({u, v});
    }

    std::vector<std::vector<int>> fragments;
    std::vector<bool> p1_visited(p1.size(), false);

    // Identify common segments in p1
    for(size_t i = 0; i < p1.size(); ++i) {
        if(p1_visited[i]) continue;
        
        // Start a new fragment with node i
        std::vector<int> current_fragment;
        current_fragment.push_back(p1[i]);
        p1_visited[i] = true;

        // Try to extend forward in P1 order
        size_t curr_idx = i;
        while(true) {
            size_t next_idx = (curr_idx + 1) % p1.size();
            int u = p1[curr_idx];
            int v = p1[next_idx];
            int u_s = u, v_s = v;
            if(u_s > v_s) std::swap(u_s, v_s);

            // Check if edge is common
            if(p2_edges.count({u_s, v_s})) {
                if(p1_visited[next_idx] && next_idx != i) {
                     break;
                }
                if(next_idx == i) {
                    break;
                }
                current_fragment.push_back(v);
                p1_visited[next_idx] = true;
                curr_idx = next_idx;
            } else {
                break;
            }
        }
        fragments.push_back(current_fragment);
    }

    // Now check if any common nodes were missed (isolated common nodes)
    std::set<int> p2_nodes(p2.begin(), p2.end());
    for(size_t i=0; i<p1.size(); ++i) {
        int u = p1[i];
        if (p2_nodes.count(u)) {
            // Check if u is already in a fragment
            bool in_frag = false;
            for(const auto& frag : fragments) { for(int node : frag) if(node == u) { in_frag = true; break; } if(in_frag) break; }
            if(!in_frag) {
                 fragments.push_back({u});
            }
        }
    }

    // If total nodes < solution_length, add random nodes
    std::set<int> current_nodes;
    for(const auto& frag : fragments) 
        for(int node : frag) current_nodes.insert(node);
    
    std::vector<int> available_nodes;
    for(int i=0; i<points_length; ++i) {
        if(!current_nodes.count(i)) available_nodes.push_back(i);
    }
    std::shuffle(available_nodes.begin(), available_nodes.end(), rng);

    while(current_nodes.size() < (size_t)solution_length && !available_nodes.empty()) {
        int new_node = available_nodes.back();
        available_nodes.pop_back();
        current_nodes.insert(new_node);
        fragments.push_back({new_node});
    }

    // Connect fragments randomly
    std::shuffle(fragments.begin(), fragments.end(), rng);
    Solution child;
    child.reserve(solution_length);

    for(auto& frag : fragments) {
        // Randomly flip fragment
        if(std::uniform_int_distribution<int>(0, 1)(rng)) {
            std::reverse(frag.begin(), frag.end());
        }
        child.insert(child.end(), frag.begin(), frag.end());
    }

    // If we have more than solution_length (unlikely unless p1/p2 mismatch size or logic error), trim
    if(child.size() > (size_t)solution_length) child.resize(solution_length);
    
    return child;
}

// Operator 2: P1 filtered by P2 + Repair
inline Solution hea_operator_2(const Solution& p1, const Solution& p2, int solution_length, int points_length, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs) {
    std::set<int> p2_nodes(p2.begin(), p2.end());
    
    // "Remove from this solution [p1] all ... nodes that are not present in the other parent [p2]."
    // "It will preserve not only common subpaths, but also order and traversal directions of the from one of the parents."
    
    // We strictly preserve the order of nodes as they appear in p1, keeping only those found in p2.
    Solution child;
    child.reserve(solution_length);
    
    for (int node : p1) {
        if (p2_nodes.count(node)) {
            child.push_back(node);
        }
    }

    // Repair if the solution is too short
    if (child.size() < (size_t)solution_length) {
         child = nn_insert_weighted_regret(std::move(child), solution_length, distance_mat, node_costs);
    }
    
    return child;
}

enum class HEAVariant {
    Op1_LS,
    Op2_NoLS,
    Op2_LS
};

inline auto hybrid_evolutionary_algorithm(
    std::chrono::nanoseconds stop_duration,
    const DistanceMatrix& distance_mat,
    const std::vector<int>& node_costs,
    int solution_length,
    int points_length,
    auto& random_engine,
    HEAVariant variant
) {
    auto start_time = std::chrono::steady_clock::now();
    size_t ls_runs = 0;
    
    std::vector<Solution> population;
    std::vector<int> pop_scores;
    const size_t POP_SIZE = 20;
    
    while(population.size() < POP_SIZE) {
        Solution sol = generate_random_solution(solution_length, points_length, random_engine);
        sol = local_search_steepest_edges(sol, distance_mat, node_costs);
        int score = calculate_objective_function(distance_mat, sol, node_costs);
        
        bool exists = false;
        for(int s : pop_scores) if(s == score) { exists = true; break; }
        
        if(!exists) {
            population.push_back(sol);
            pop_scores.push_back(score);
            ls_runs++;
        }
    }
    
    Solution best_solution = population[0];
    int best_score = pop_scores[0];
    for(size_t i=1; i<POP_SIZE; ++i) {
        if(pop_scores[i] < best_score) {
            best_score = pop_scores[i];
            best_solution = population[i];
        }
    }
    
    std::uniform_int_distribution<size_t> parent_dist(0, POP_SIZE - 1);
    
    while(std::chrono::steady_clock::now() - start_time < stop_duration) {
        size_t idx1 = parent_dist(random_engine);
        size_t idx2 = parent_dist(random_engine);
        while(idx1 == idx2) idx2 = parent_dist(random_engine); 
        
        const Solution& p1 = population[idx1];
        const Solution& p2 = population[idx2];
        
        Solution child;
        if(variant == HEAVariant::Op1_LS) {
            child = hea_operator_1(p1, p2, solution_length, points_length, random_engine, distance_mat, node_costs);
            child = local_search_steepest_edges(std::move(child), distance_mat, node_costs);
            ls_runs++;
        } else if (variant == HEAVariant::Op2_NoLS) {
            child = hea_operator_2(p1, p2, solution_length, points_length, distance_mat, node_costs);
        } else { // Op2_LS
            child = hea_operator_2(p1, p2, solution_length, points_length, distance_mat, node_costs);
            child = local_search_steepest_edges(std::move(child), distance_mat, node_costs);
            ls_runs++;
        }
        
        int child_score = calculate_objective_function(distance_mat, child, node_costs);
        
        if(child_score < best_score) {
            best_score = child_score;
            best_solution = child;
        }
        
        bool exists = false;
        for(int s : pop_scores) if(s == child_score) { exists = true; break; }
        
        if(!exists) {
            int max_score = -1;
            size_t worst_idx = 0;
            for(size_t i=0; i<POP_SIZE; ++i) {
                if(pop_scores[i] > max_score) {
                    max_score = pop_scores[i];
                    worst_idx = i;
                }
            }
            
            if(child_score < max_score) {
                population[worst_idx] = std::move(child);
                pop_scores[worst_idx] = child_score;
            }
        }
    }
    
    return std::make_tuple(best_solution, ls_runs);
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

    auto rng = std::mt19937{156053 + 156042};

    int solution_length = points.size() / 2; // - 1
    
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

    // 2. ILS
    std::cout << "Running ILS" << std::flush;
    for (size_t i = 0; i < max_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        const auto result{ iterated_local_search(avg_ns, distance_mat, node_costs, solution_length, points.size(), rng) };
        solutions.emplace_back(std::get<0>(result));
        ls_runs.emplace_back(std::get<1>(result));
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(end - start);
        std::cout << "\rRunning ILS " << ProgressBar(i, max_runs, max_runs) << " " << std::chrono::duration_cast<std::chrono::seconds>(times.back()).count() * (max_runs - i - 1) << "s left     " << std::flush;
    }
    std::cout << std::endl;
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ils", "ils");
    calculate_and_print_time_statistics(times);
    print_ls_runs_staticits(ls_runs);
    solutions.clear(); ls_runs.clear(); times.clear(); std::cout << std::endl;

    // 3. LNS (with LS)
    std::cout << "Running LNS (with ls)" << std::flush;
    for (size_t i = 0; i < max_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        const auto result{ large_neighborhood_search<true>(avg_ns, distance_mat, node_costs, solution_length, points.size(), rng) };
        solutions.emplace_back(std::get<0>(result));
        ls_runs.emplace_back(std::get<1>(result));
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(end - start);
        std::cout << "\rRunning LNS (with ls) " << ProgressBar(i, max_runs, max_runs) << " " << std::chrono::duration_cast<std::chrono::seconds>(times.back()).count() * (max_runs - i - 1) << "s left     " << std::flush;
    }
    std::cout << std::endl;
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "lns_ls", "lns_ls");
    calculate_and_print_time_statistics(times);
    print_ls_runs_staticits(ls_runs);
    solutions.clear(); ls_runs.clear(); times.clear(); std::cout << std::endl;

    // 4. HEA Op1 (with LS)
    std::cout << "Running HEA Op1 (with ls)" << std::flush;
    for (size_t i = 0; i < max_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        const auto result{ hybrid_evolutionary_algorithm(avg_ns, distance_mat, node_costs, solution_length, points.size(), rng, HEAVariant::Op1_LS) };
        solutions.emplace_back(std::get<0>(result));
        ls_runs.emplace_back(std::get<1>(result));
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(end - start);
        std::cout << "\rRunning HEA Op1 (with ls) " << ProgressBar(i, max_runs, max_runs) << " " << std::chrono::duration_cast<std::chrono::seconds>(times.back()).count() * (max_runs - i - 1) << "s left     " << std::flush;
    }
    std::cout << std::endl;
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "hea_op1", "hea_op1");
    calculate_and_print_time_statistics(times);
    print_ls_runs_staticits(ls_runs);
    solutions.clear(); ls_runs.clear(); times.clear(); std::cout << std::endl;
    
    // 5. HEA Op2 (no LS)
    std::cout << "Running HEA Op2 (no ls)" << std::flush;
    for (size_t i = 0; i < max_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        const auto result{ hybrid_evolutionary_algorithm(avg_ns, distance_mat, node_costs, solution_length, points.size(), rng, HEAVariant::Op2_NoLS) };
        solutions.emplace_back(std::get<0>(result));
        ls_runs.emplace_back(std::get<1>(result));
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(end - start);
        std::cout << "\rRunning HEA Op2 (no ls) " << ProgressBar(i, max_runs, max_runs) << " " << std::chrono::duration_cast<std::chrono::seconds>(times.back()).count() * (max_runs - i - 1) << "s left     " << std::flush;
    }
    std::cout << std::endl;
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "hea_op2_nols", "hea_op2_nols");
    calculate_and_print_time_statistics(times);
    print_ls_runs_staticits(ls_runs);
    solutions.clear(); ls_runs.clear(); times.clear(); std::cout << std::endl;
    
    // 6. HEA Op2 (with LS)
    std::cout << "Running HEA Op2 (with ls)" << std::flush;
    for (size_t i = 0; i < max_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        const auto result{ hybrid_evolutionary_algorithm(avg_ns, distance_mat, node_costs, solution_length, points.size(), rng, HEAVariant::Op2_LS) };
        solutions.emplace_back(std::get<0>(result));
        ls_runs.emplace_back(std::get<1>(result));
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(end - start);
        std::cout << "\rRunning HEA Op2 (with ls) " << ProgressBar(i, max_runs, max_runs) << " " << std::chrono::duration_cast<std::chrono::seconds>(times.back()).count() * (max_runs - i - 1) << "s left     " << std::flush;
    }
    std::cout << std::endl;
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "hea_op2_ls", "hea_op2_ls");
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