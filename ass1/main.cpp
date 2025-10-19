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
        const Point& to = points[solution[(i+1) % solution.size()]];

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
    std::cout << method << " & " << min << " & " << avg << " & " << max << " \\\\";
#endif

    if (best.has_value())
        export_solution(points, *best, std::string(method_short), instance);

    std::cout << std::endl;
}

// SOLUTIONS
inline auto random_solution(int points_length, int solution_length, auto& random_engine) {
    Solution solution(points_length);
    std::iota(solution.begin(), solution.end(), 0);
    std::shuffle(solution.begin(), solution.end(), random_engine);
    solution.resize(solution_length);

    return solution;
}

inline auto nearest_neighbor_back(int solution_length, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, int starting_point) {
    Solution solution;
    solution.reserve(solution_length);
    solution.push_back(starting_point);

    while (solution.size() < solution_length) {
        const auto& neighbors = distance_mat[solution.back()];

        int min_dist = std::numeric_limits<int>::max();
        int selected = -1;
        for (size_t i = 0; i < neighbors.size(); ++i)
        {
            if (std::find(solution.begin(), solution.end(), i) != solution.end())
                continue;
            if (neighbors[i] + node_costs[i] <= min_dist)
            {
                min_dist = neighbors[i] + node_costs[i];
                selected = i;
            }
        }

        if (selected != -1) {
            solution.push_back(selected);
        } else {
            break; // No unvisited node found, break to avoid infinite loop
        }
    };

    return solution;
}

inline auto nearest_neighbor_insert(int solution_length, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, int starting_point) {
    Solution solution;
    solution.reserve(solution_length);
    solution.push_back(starting_point);

    std::vector<bool> visited(distance_mat.size(), false);
    visited[starting_point] = true;

    while (solution.size() < solution_length) {
        int best_point_to_add = -1;
        size_t best_insertion_index = -1;
        int min_cost_increase = std::numeric_limits<int>::max();

        for (size_t p = 0; p < distance_mat.size(); ++p) {
            if (!visited[p]) {
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

                    if (cost_increase < min_cost_increase) {
                        min_cost_increase = cost_increase;
                        best_point_to_add = p;
                        best_insertion_index = insert_index;
                    }
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

inline auto greedy_cycle(int solution_length, const DistanceMatrix& distance_mat, const std::vector<int>& node_costs, int starting_point) {
    Solution solution;
    solution.reserve(solution_length);
    size_t num_points = distance_mat.size();

    if (num_points < 2 || solution_length < 2) {
        if (num_points >= 1 && solution_length >= 1) solution.push_back(starting_point);
        return solution;
    }

    // 1. Initialization: Start with the given point and find its nearest neighbor to form a cycle.
    int nearest_neighbor = -1;
    int min_dist = std::numeric_limits<int>::max();
    for (size_t i = 0; i < num_points; ++i) {
        if (i != starting_point) {
            int dist = distance_mat[starting_point][i] + distance_mat[i][starting_point] + node_costs[i];
            if (dist < min_dist) {
                min_dist = dist;
                nearest_neighbor = i;
            }
        }
    }
    
    solution.push_back(starting_point);
    solution.push_back(nearest_neighbor);
    
    std::vector<bool> visited(num_points, false);
    visited[starting_point] = true;
    visited[nearest_neighbor] = true;

    // 2. Iterative Expansion with weighted regret
    while (solution.size() < solution_length) {
        int best_point_to_add = -1;
        size_t best_insertion_index = -1;
        double best_min_cost = std::numeric_limits<double>::max();

        for (size_t p = 0; p < num_points; ++p) {
            if (!visited[p]) {
                int min_cost = std::numeric_limits<int>::max();
                size_t current_best_insertion_index = -1;

                for (size_t j = 0; j < solution.size(); ++j) {
                    int from_node = solution[j];
                    int to_node = solution[(j + 1) % solution.size()];
                    int cost_increase = distance_mat[from_node][p] + distance_mat[p][to_node] - distance_mat[from_node][to_node] + node_costs[p];

                    if (cost_increase < min_cost) {
                        min_cost = cost_increase;
                        current_best_insertion_index = j + 1;
                    }
                }

                if (min_cost < best_min_cost) {
                    best_min_cost = min_cost;
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

    int solution_length = points.size() / 2; // - 1
    std::cout << std::endl;

    auto rng = std::default_random_engine{};
    rng.seed(156053 + 156042);
    
    std::vector<int> node_costs;
    node_costs.reserve(points.size());
    for(const auto& p : points) {
        node_costs.push_back(p.cost);
    }

    Solutions solutions;
    for (size_t i = 0; i < points.size(); ++i) 
        solutions.emplace_back(random_solution(points.size(), solution_length, rng));
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "random", "Random solution");

    solutions.clear();
    for (size_t i = 0; i < points.size(); ++i)
        solutions.emplace_back(nearest_neighbor_back(solution_length, distance_mat, node_costs, i));
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "nn_back", "Nearest neighbor considering adding the node only at the end of the current path");
    
    solutions.clear();
    for (size_t i = 0; i < points.size(); ++i)
        solutions.emplace_back(nearest_neighbor_insert(solution_length, distance_mat, node_costs, i));
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "nn_insert", "Nearest neighbor considering adding the node at all possible position");

    solutions.clear();
    for (size_t i = 0; i < points.size(); ++i)
        solutions.emplace_back(greedy_cycle(solution_length, distance_mat, node_costs, i));
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "gc", "Greedy Cycle");

    return 0;
}