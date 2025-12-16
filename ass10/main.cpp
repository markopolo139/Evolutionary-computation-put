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

inline size_t tournament_selection(const std::vector<int>& scores, auto& rng, int k = 2) {
    std::uniform_int_distribution<size_t> dist(0, scores.size() - 1);
    size_t best_idx = dist(rng);
    int best_score = scores[best_idx];
    
    for (int i = 1; i < k; ++i) {
        size_t next_idx = dist(rng);
        int next_score = scores[next_idx];
        if (next_score < best_score) {
            best_score = next_score;
            best_idx = next_idx;
        }
    }
    return best_idx;
}

inline auto improved_hybrid_evolutionary_algorithm(
    std::chrono::nanoseconds stop_duration,
    const DistanceMatrix& distance_mat,
    const std::vector<int>& node_costs,
    int solution_length,
    int points_length,
    auto& random_engine
) {
    auto start_time = std::chrono::steady_clock::now();
    size_t ls_runs = 0;
    
    // Island Model Configuration
    const int NUM_ISLANDS = 6;
    const int POP_SIZE_PER_ISLAND = 20;
    const int MIGRATION_INTERVAL = 50; 
    const int STAGNATION_LIMIT = 250; 

    using Population = std::vector<Solution>;
    using PopScores = std::vector<int>;

    std::vector<Population> islands(NUM_ISLANDS);
    std::vector<PopScores> islands_scores(NUM_ISLANDS);
    std::vector<int> islands_stagnation(NUM_ISLANDS, 0);

    // Helper to generate a new individual
    auto create_individual = [&]() {
        Solution sol = generate_random_solution(solution_length, points_length, random_engine);
        sol = local_search_steepest_edges(std::move(sol), distance_mat, node_costs);
        return sol;
    };

    // Initialization
    for (int i = 0; i < NUM_ISLANDS; ++i) {
        while (islands[i].size() < POP_SIZE_PER_ISLAND) {
            Solution sol = create_individual();
            ls_runs++;
            int score = calculate_objective_function(distance_mat, sol, node_costs);

            bool exists = false;
            for (int s : islands_scores[i]) if (s == score) { exists = true; break; }

            if (!exists) {
                islands[i].push_back(sol);
                islands_scores[i].push_back(score);
            }
        }
    }

    // Global Best Tracking
    Solution best_solution = islands[0][0];
    int best_score = islands_scores[0][0];
    
    auto update_global_best = [&](const Solution& sol, int score) {
        if (score < best_score) {
            best_score = score;
            best_solution = sol;
        }
    };

    // Initial check
    for(int i=0; i<NUM_ISLANDS; ++i) {
        for(size_t k=0; k<islands_scores[i].size(); ++k) {
            update_global_best(islands[i][k], islands_scores[i][k]);
        }
    }

    int iteration_count = 0;

    while (std::chrono::steady_clock::now() - start_time < stop_duration) {
        iteration_count++;

        // Evolve each island
        for (int i = 0; i < NUM_ISLANDS; ++i) {
            // Check Stagnation
            if (islands_stagnation[i] > STAGNATION_LIMIT) {
                // Restart Island: Keep Best, Randomize Rest
                // Find best in island
                int best_island_score = std::numeric_limits<int>::max();
                size_t best_idx = 0;
                for(size_t k=0; k<islands_scores[i].size(); ++k) {
                    if(islands_scores[i][k] < best_island_score) {
                        best_island_score = islands_scores[i][k];
                        best_idx = k;
                    }
                }
                Solution elite = islands[i][best_idx]; // Copy elite
                
                // Clear and rebuild
                islands[i].clear();
                islands_scores[i].clear();
                
                // Add elite back
                islands[i].push_back(elite);
                islands_scores[i].push_back(best_island_score);
                
                // Fill rest
                while (islands[i].size() < POP_SIZE_PER_ISLAND) {
                    Solution sol = create_individual();
                    ls_runs++;
                    int score = calculate_objective_function(distance_mat, sol, node_costs);
                    
                    bool exists = false;
                    for (int s : islands_scores[i]) if (s == score) { exists = true; break; }

                    if (!exists) {
                        islands[i].push_back(sol);
                        islands_scores[i].push_back(score);
                    }
                }
                
                islands_stagnation[i] = 0; // Reset counter
            }

            // Selection (Tournament)
            size_t idx1 = tournament_selection(islands_scores[i], random_engine);
            size_t idx2 = tournament_selection(islands_scores[i], random_engine);
            while (idx1 == idx2 && islands[i].size() > 1) {
                idx2 = tournament_selection(islands_scores[i], random_engine);
            }

            const Solution& p1 = islands[i][idx1];
            const Solution& p2 = islands[i][idx2];

            // Recombination (Op1 + LS) -> Using Op1 as it performed better in baseline
            Solution child = hea_operator_1(p1, p2, solution_length, points_length, random_engine, distance_mat, node_costs);
            child = local_search_steepest_edges(std::move(child), distance_mat, node_costs);
            ls_runs++;

            int child_score = calculate_objective_function(distance_mat, child, node_costs);

            // Update Global Best
            update_global_best(child, child_score);

            // Replacement (Steady State)
            bool improved_island = false;
            
            // Check existence
            bool exists = false;
            for (int s : islands_scores[i]) if (s == child_score) { exists = true; break; }

            if (!exists) {
                // Replace worst
                int max_score = -1;
                size_t worst_idx = 0;
                for (size_t k = 0; k < islands_scores[i].size(); ++k) {
                    if (islands_scores[i][k] > max_score) {
                        max_score = islands_scores[i][k];
                        worst_idx = k;
                    }
                }

                if (child_score < max_score) {
                    islands[i][worst_idx] = std::move(child);
                    islands_scores[i][worst_idx] = child_score;
                    improved_island = true;
                }
            }
            
            if (improved_island) {
                islands_stagnation[i] = 0;
            } else {
                islands_stagnation[i]++;
            }
        }

        // Migration
        if (iteration_count % MIGRATION_INTERVAL == 0) {
            for (int i = 0; i < NUM_ISLANDS; ++i) {
                int target_island = (i + 1) % NUM_ISLANDS;

                // Find best in source
                int best_src_score = std::numeric_limits<int>::max();
                size_t best_src_idx = 0;
                for (size_t k = 0; k < islands_scores[i].size(); ++k) {
                    if (islands_scores[i][k] < best_src_score) {
                        best_src_score = islands_scores[i][k];
                        best_src_idx = k;
                    }
                }

                Solution migrant = islands[i][best_src_idx];
                int migrant_score = best_src_score;

                // Check existence in target
                bool exists = false;
                for (int s : islands_scores[target_island]) if (s == migrant_score) { exists = true; break; }

                if (!exists) {
                    // Find worst in target
                    int max_tgt_score = -1;
                    size_t worst_tgt_idx = 0;
                    for (size_t k = 0; k < islands_scores[target_island].size(); ++k) {
                        if (islands_scores[target_island][k] > max_tgt_score) {
                            max_tgt_score = islands_scores[target_island][k];
                            worst_tgt_idx = k;
                        }
                    }

                    if (migrant_score < max_tgt_score) {
                        islands[target_island][worst_tgt_idx] = migrant;
                        islands_scores[target_island][worst_tgt_idx] = migrant_score;
                        // Migration counts as improvement to delay stagnation
                        islands_stagnation[target_island] = 0; 
                    }
                }
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
    auto stop_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(3000));

    // Improved HEA (Islands + Roulette)
    std::cout << "Running Improved HEA" << std::flush;
    for (size_t i = 0; i < max_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        const auto result{ improved_hybrid_evolutionary_algorithm(stop_duration, distance_mat, node_costs, solution_length, points.size(), rng) };
        solutions.emplace_back(std::get<0>(result));
        ls_runs.emplace_back(std::get<1>(result));
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(end - start);
        std::cout << "\rRunning Improved HEA " << ProgressBar(i, max_runs, max_runs) << " " << std::chrono::duration_cast<std::chrono::seconds>(times.back()).count() * (max_runs - i - 1) << "s left     " << std::flush;
    }
    std::cout << std::endl;
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "improved_hea", "improved_hea");
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