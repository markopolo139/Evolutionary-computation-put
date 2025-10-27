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

// SOLUTIONS
Solution local_search_steepest_edges(Solution solution, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, int max_steps = 10000) {
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

inline auto calculate_candidate_list(const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, int num_candidates) {
    int num_nodes = distance_mat.size();
    std::vector<std::vector<int>> candidate_list(num_nodes);

    for (int i = 0; i < num_nodes; ++i) {
        std::vector<std::pair<int, int>> sorted_neighbors;
        for (int j = 0; j < num_nodes; ++j) {
            if (i == j) continue;
            int cost = distance_mat[i][j] + node_costs[j];
            sorted_neighbors.push_back({cost, j});
        }

        std::sort(sorted_neighbors.begin(), sorted_neighbors.end());

        candidate_list[i].reserve(num_candidates);
        for (int k = 0; k < num_candidates && k < sorted_neighbors.size(); ++k) {
            candidate_list[i].push_back(sorted_neighbors[k].second);
        }
    }
    return candidate_list;
}

inline bool is_candidate(int u, int v, const std::vector<std::vector<int>>& candidate_list) {
    const auto& u_candidates = candidate_list.at(u);
    if (std::find(u_candidates.begin(), u_candidates.end(), v) != u_candidates.end()) {
        return true;
    }
    
    const auto& v_candidates = candidate_list.at(v);
    if (std::find(v_candidates.begin(), v_candidates.end(), u) != v_candidates.end()) {
        return true;
    }

    return false;
}

Solution local_search_candidate_moves(Solution solution, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, const std::vector<std::vector<int>>& candidate_list, int max_steps = 10000) {
    bool improvement = true;
    int steps = 0;
    while (improvement && steps < max_steps) {
        improvement = false;
        steps++;
        int best_delta = 0;
        std::function<void()> best_move;

        std::map<int, int> solution_node_to_idx;
        for(size_t i=0; i<solution.size(); ++i) {
            solution_node_to_idx[solution[i]] = i;
        }

        // Intra-route moves (edge exchange)
        for (size_t i = 0; i < solution.size(); ++i) {
            int u1 = solution[i];
            int v1 = solution[(i + 1) % solution.size()];

            for (int u2 : candidate_list[u1]) {
                auto it = solution_node_to_idx.find(u2);
                if (it == solution_node_to_idx.end()) continue;
                size_t j = it->second;

                if (i >= j) continue;

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
        std::vector<bool> in_solution(distance_mat.size(), false);
        for (int node : solution) in_solution[node] = true;

        for (size_t i = 0; i < solution.size(); ++i) {
            int current_node = solution[i];
            int prev = (i == 0) ? solution.back() : solution[i - 1];
            int next = (i == solution.size() - 1) ? solution.front() : solution[i + 1];

            std::vector<int> candidates_to_check;
            candidates_to_check.insert(candidates_to_check.end(), candidate_list[prev].begin(), candidate_list[prev].end());
            if(candidate_list.size() > next)
                candidates_to_check.insert(candidates_to_check.end(), candidate_list[next].begin(), candidate_list[next].end());
            std::sort(candidates_to_check.begin(), candidates_to_check.end());
            candidates_to_check.erase(std::unique(candidates_to_check.begin(), candidates_to_check.end()), candidates_to_check.end());


            for (int non_sol_node : candidates_to_check) {
                if (in_solution[non_sol_node]) continue;

                int delta = (distance_mat[prev][non_sol_node] + distance_mat[non_sol_node][next] + node_costs[non_sol_node]) -
                            (distance_mat[prev][current_node] + distance_mat[current_node][next] + node_costs[current_node]);

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

    auto rng = std::default_random_engine{};
    rng.seed(156053 + 156042);

    int solution_length = points.size() / 2; // - 1
    std::cout << std::endl;
    
    Solutions solutions;
    std::vector<std::chrono::duration<double, std::milli>> times;

    for (size_t i = 0; i < 100; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        solutions.emplace_back(local_search_steepest_edges(generate_random_solution(solution_length, points.size(), rng), distance_mat, node_costs));
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(end - start);
    }
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ls_steepest_edges_random", "ls_steepest_edges_random");
    calculate_and_print_time_statistics(times);
    solutions.clear();
    times.clear();
    std::cout << std::endl;

    auto candidate_list = calculate_candidate_list(distance_mat, node_costs, 10);
    std::cout << "calculated candidate list" << std::endl;

    for (size_t i = 0; i < 100; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        solutions.emplace_back(local_search_candidate_moves(generate_random_solution(solution_length, points.size(), rng), distance_mat, node_costs, candidate_list));
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(end - start);
    }
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ls_candidate_random", "ls_candidate_random");
    calculate_and_print_time_statistics(times);
    solutions.clear();
    times.clear();
    std::cout << std::endl;

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
