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
    std::cout << "Min:    " << min << std::endl;
    std::cout << "Avg:    " << avg << std::endl;
    std::cout << "Max:    " << max << std::endl;
#else
    std::cout << method << " & " << min << " & " << avg << " & " << max << " \\";
#endif

    if (best.has_value())
        export_solution(points, *best, std::string(method_short), instance);

    std::cout << std::endl;
}

#include <unordered_set>
// SOLUTIONS

// starting solutions
// - random
// - nn insert weighted regret

// local search - type of greedy, random changes
// local search - type of neighborhood, intra/inter moves, (for intra: two-nodes exchange/two-edges exchange)

inline auto local_search_2_opt(Solution solution, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, int max_steps = 1000) {
    bool improvement = true;
    int steps = 0;
    int n = solution.size();
    while (improvement && steps < max_steps) {
        improvement = false;
        steps++;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                int u1 = solution[i];
                int v1 = solution[(i + 1) % n];
                int u2 = solution[j];
                int v2 = solution[(j + 1) % n];

                if (v1 == u2 || u1 == v2) continue;

                int current_dist = distance_mat[u1][v1] + distance_mat[u2][v2];
                int new_dist = distance_mat[u1][u2] + distance_mat[v1][v2];

                if (new_dist < current_dist) {
                    std::reverse(solution.begin() + i + 1, solution.begin() + j + 1);
                    improvement = true;
                }
            }
        }
    }
    return solution;
}

inline auto nn_insert_weighted_regret(int solution_length, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, int starting_point, double regret_weight = 1.0, double cost_weight = 1.0) {
    Solution solution;
    solution.reserve(solution_length);
    solution.push_back(starting_point);

    std::vector<bool> visited(distance_mat.size(), false);
    visited[starting_point] = true;

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

Solution local_search_steepest_nodes(Solution solution, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, int max_steps = 1000) {
    bool improvement = true;
    int steps = 0;
    while (improvement && steps < max_steps) {
        improvement = false;
        steps++;
        int best_delta = 0;
        std::function<void()> best_move;

        // Intra-route moves (node exchange)
        for (size_t i = 0; i < solution.size(); ++i) {
            for (size_t j = i + 1; j < solution.size(); ++j) {
                int prev_i = (i == 0) ? solution.back() : solution[i - 1];
                int next_i = (i == solution.size() - 1) ? solution.front() : solution[i + 1];
                int prev_j = (j == 0) ? solution.back() : solution[j - 1];
                int next_j = (j == solution.size() - 1) ? solution.front() : solution[j + 1];

                int delta;
                if (j == i + 1) {
                    delta = distance_mat[prev_i][solution[j]] + distance_mat[solution[i]][next_j] -
                            (distance_mat[prev_i][solution[i]] + distance_mat[solution[j]][next_j]);
                } else {
                    delta = distance_mat[prev_i][solution[j]] + distance_mat[solution[j]][next_i] +
                            distance_mat[prev_j][solution[i]] + distance_mat[solution[i]][next_j] -
                            (distance_mat[prev_i][solution[i]] + distance_mat[solution[i]][next_i] +
                             distance_mat[prev_j][solution[j]] + distance_mat[solution[j]][next_j]);
                }

                if (delta < best_delta) {
                    best_delta = delta;
                    best_move = [=, &solution]() {
                        std::swap(solution[i], solution[j]);
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

Solution local_search_steepest_edges(Solution solution, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, int max_steps = 1000) {
    bool improvement = true;
    int steps = 0;
    while (improvement && steps < max_steps) {
        improvement = false;
        steps++;
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

Solution local_search_greedy_nodes(Solution solution, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, int max_steps = 1000) {
    bool improvement = true;
    std::random_device rd;
    std::mt19937 g(rd());
    int steps = 0;
    while (improvement && steps < max_steps) {
        improvement = false;
        steps++;
        
        std::vector<std::function<void()>> moves;

        // Intra-route moves (node exchange)
        for (size_t i = 0; i < solution.size(); ++i) {
            for (size_t j = i + 1; j < solution.size(); ++j) {
                moves.push_back([=, &solution, &improvement](){
                    int prev_i = (i == 0) ? solution.back() : solution[i - 1];
                    int next_i = (i == solution.size() - 1) ? solution.front() : solution[i + 1];
                    int prev_j = (j == 0) ? solution.back() : solution[j - 1];
                    int next_j = (j == solution.size() - 1) ? solution.front() : solution[j + 1];

                    int delta;
                    if (j == i + 1) {
                        delta = distance_mat[prev_i][solution[j]] + distance_mat[solution[i]][next_j] -
                                (distance_mat[prev_i][solution[i]] + distance_mat[solution[j]][next_j]);
                    } else {
                        delta = distance_mat[prev_i][solution[j]] + distance_mat[solution[j]][next_i] +
                                distance_mat[prev_j][solution[i]] + distance_mat[solution[i]][next_j] -
                                (distance_mat[prev_i][solution[i]] + distance_mat[solution[i]][next_i] +
                                 distance_mat[prev_j][solution[j]] + distance_mat[solution[j]][next_j]);
                    }

                    if (delta < 0) {
                        std::swap(solution[i], solution[j]);
                        improvement = true;
                    }
                });
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
                moves.push_back([=, &solution, &improvement](){
                    int prev = (i == 0) ? solution.back() : solution[i - 1];
                    int next = (i == solution.size() - 1) ? solution.front() : solution[i + 1];
                    
                    int delta = (distance_mat[prev][non_sol_node] + distance_mat[non_sol_node][next] + node_costs[non_sol_node]) -
                                (distance_mat[prev][solution[i]] + distance_mat[solution[i]][next] + node_costs[solution[i]]);

                    if (delta < 0) {
                        solution[i] = non_sol_node;
                        improvement = true;
                    }
                });
            }
        }
        
        std::shuffle(moves.begin(), moves.end(), g);

        for(auto& move : moves) {
            move();
            if(improvement) break;
        }
    }
    return solution;
}

Solution local_search_greedy_edges(Solution solution, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, int max_steps = 1000) {
    bool improvement = true;
    std::random_device rd;
    std::mt19937 g(rd());
    int steps = 0;
    while (improvement && steps < max_steps) {
        improvement = false;
        steps++;
        std::vector<std::function<void()>> moves;

        // Intra-route moves (edge exchange)
        for (size_t i = 0; i < solution.size(); ++i) {
            for (size_t j = i + 1; j < solution.size(); ++j) {
                moves.push_back([=, &solution, &improvement](){
                    int u1 = solution[i];
                    int v1 = solution[(i + 1) % solution.size()];
                    int u2 = solution[j];
                    int v2 = solution[(j + 1) % solution.size()];
                    if (v1 == u2 || u1 == v2) return;

                    int delta = distance_mat[u1][u2] + distance_mat[v1][v2] -
                                (distance_mat[u1][v1] + distance_mat[u2][v2]);

                    if (delta < 0) {
                        std::reverse(solution.begin() + i + 1, solution.begin() + j + 1);
                        improvement = true;
                    }
                });
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
                moves.push_back([=, &solution, &improvement](){
                    int prev = (i == 0) ? solution.back() : solution[i - 1];
                    int next = (i == solution.size() - 1) ? solution.front() : solution[i + 1];
                    
                    int delta = (distance_mat[prev][non_sol_node] + distance_mat[non_sol_node][next] + node_costs[non_sol_node]) -
                                (distance_mat[prev][solution[i]] + distance_mat[solution[i]][next] + node_costs[solution[i]]);

                    if (delta < 0) {
                        solution[i] = non_sol_node;
                        improvement = true;
                    }
                });
            }
        }
        
        std::shuffle(moves.begin(), moves.end(), g);

        for(auto& move : moves) {
            move();
            if(improvement) break;
        }
    }
    return solution;
}

Solution generate_random_solution(int solution_length, int num_points) {
    Solution solution;
    solution.reserve(solution_length);
    std::vector<int> p(num_points);
    std::iota(p.begin(), p.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(p.begin(), p.end(), g);
    for (int i = 0; i < solution_length; ++i) {
        solution.push_back(p[i]);
    }
    return solution;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Error: This program requires exactly one argument. It should specify the name of the instance" << std::endl;
        std::cerr << "Usage: " << argv[0] << " <instance>" << std::endl;
        return 1; 
    }
    std::string instance = argv[1];
    std::cout << "results for: " << instance << std::endl;

    std::string instance_file = instance + ".csv";
    std::ifstream file(instance_file);
    if (!file.is_open()) {
        std::cerr << "Error: File not found: " << instance_file << std::endl;
        return 1;
    }

    auto points = read_points(file);
    if (points.empty()){
        std::cerr << "Error: No points found in the file: " << instance_file<< std::endl;
        return 1;
    }
    std::cout << "collected points: " << points.size() << std::endl;

    auto distance_mat = calculate_distance_matrix(points);
    std::cout << "calculated the distance matrix" << std::endl;

    std::vector<int> node_costs;
    node_costs.reserve(points.size());
    for(const auto& p : points) {
        node_costs.push_back(p.cost);
    }

    int solution_length = points.size() / 2; // - 1
    std::cout << std::endl;
    
    Solutions solutions;

    // 1. Steepest nodes with random start
    for (size_t i = 0; i < 10; ++i) {
        solutions.emplace_back(local_search_steepest_nodes(generate_random_solution(solution_length, points.size()), distance_mat, node_costs));
    }
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ls_steepest_nodes_random", "ls_steepest_nodes_random");
    solutions.clear();

    // 2. Steepest nodes with greedy start
    for (size_t i = 0; i < points.size(); ++i) {
        solutions.emplace_back(local_search_steepest_nodes(nn_insert_weighted_regret(solution_length, distance_mat, node_costs, i), distance_mat, node_costs));
    }
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ls_steepest_nodes_greedy", "ls_steepest_nodes_greedy");
    solutions.clear();

    // 3. Steepest edges with random start
    for (size_t i = 0; i < 100; ++i) {
        solutions.emplace_back(local_search_steepest_edges(generate_random_solution(solution_length, points.size()), distance_mat, node_costs));
    }
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ls_steepest_edges_random", "ls_steepest_edges_random");
    solutions.clear();

    // 4. Steepest edges with greedy start
    for (size_t i = 0; i < points.size(); ++i) {
        solutions.emplace_back(local_search_steepest_edges(nn_insert_weighted_regret(solution_length, distance_mat, node_costs, i), distance_mat, node_costs));
    }
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ls_steepest_edges_greedy", "ls_steepest_edges_greedy");
    solutions.clear();

    // 5. Greedy nodes with random start
    for (size_t i = 0; i < 100; ++i) {
        solutions.emplace_back(local_search_greedy_nodes(generate_random_solution(solution_length, points.size()), distance_mat, node_costs));
    }
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ls_greedy_nodes_random", "ls_greedy_nodes_random");
    solutions.clear();

    // 6. Greedy nodes with greedy start
    for (size_t i = 0; i < points.size(); ++i) {
        solutions.emplace_back(local_search_greedy_nodes(nn_insert_weighted_regret(solution_length, distance_mat, node_costs, i), distance_mat, node_costs));
    }
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ls_greedy_nodes_greedy", "ls_greedy_nodes_greedy");
    solutions.clear();

    // 7. Greedy edges with random start
    for (size_t i = 0; i < 100; ++i) {
        solutions.emplace_back(local_search_greedy_edges(generate_random_solution(solution_length, points.size()), distance_mat, node_costs));
    }
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ls_greedy_edges_random", "ls_greedy_edges_random");
    solutions.clear();

    // 8. Greedy edges with greedy start
    for (size_t i = 0; i < points.size(); ++i) {
        solutions.emplace_back(local_search_greedy_edges(nn_insert_weighted_regret(solution_length, distance_mat, node_costs, i), distance_mat, node_costs));
    }
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ls_greedy_edges_greedy", "ls_greedy_edges_greedy");
    solutions.clear();

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
