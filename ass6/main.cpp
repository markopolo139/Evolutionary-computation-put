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

/**
 * @brief Perturbs a solution to escape a local optimum.
 * This implementation performs a "multiple swap" move:
 * It swaps 'perturbation_strength' nodes from the solution with 'perturbation_strength' nodes
 * that are *not* currently in the solution.
 *
 * @param solution The current local optimum solution.
 * @param points_length The total number of points available (N).
 * @param perturbation_strength The number of nodes to swap.
 * @param random_engine The random number generator.
 * @return A new, perturbed solution.
 */
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

/**
 * @brief Iterated Local Search (ILS).
 * Starts with a local optimum, then repeatedly perturbs it and runs local search again,
 * accepting the new solution if it's better.
 *
 * @param stop_duration The maximum time to run for (e.g., the time taken by MSLS).
 * @param perturbation_strength The number of swaps to perform in each perturbation.
 * @return The best solution found.
 */
inline auto iterated_local_search(
    std::chrono::nanoseconds stop_duration,
    const DistanceMatrix& distance_mat,
    const std::vector<int>& node_costs,
    int solution_length,
    int points_length,
    auto& random_engine,
    int perturbation_strength = 3 // Default perturbation strength
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
        y = local_search_steepest_edges(y, distance_mat, node_costs); ++ls_runs;
        
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
    std::vector<size_t> ls_runs;

    constexpr size_t max_runs = 20;
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
    solutions.clear();
    times.clear();
    std::cout << std::endl;

    constexpr int perturbation_strength = 21;
    std::cout << "Running ILS (ps=" << perturbation_strength << ")" << std::flush;
    for (size_t i = 0; i < max_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        const auto result{ iterated_local_search(avg_ns, distance_mat, node_costs, solution_length, points.size(), rng, perturbation_strength) };
        solutions.emplace_back(std::get<0>(result));
        ls_runs.emplace_back(std::get<1>(result));
        auto end = std::chrono::high_resolution_clock::now();
        times.push_back(end - start);
        std::cout << "\rRunning ILS (ps=" << perturbation_strength << ") " << ProgressBar(i, max_runs, max_runs) << " " << std::chrono::duration_cast<std::chrono::seconds>(times.back()).count() * (max_runs - i - 1) << "s left     " << std::flush;
    }
    std::cout << std::endl;
    calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ils", "ils");
    calculate_and_print_time_statistics(times);
    print_ls_runs_staticits(ls_runs);
    solutions.clear();
    ls_runs.clear();
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
