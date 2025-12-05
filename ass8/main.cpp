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

using SimilarityMeasure = size_t (*)(const Solution& a, const Solution& b);

size_t similarity_edges(const Solution& a, const Solution& b) {
    // Collect edges from solution 'a'. We normalize them (min, max) 
    // because in an undirected graph (u, v) is the same as (v, u).
    std::vector<std::pair<int, int>> edges_a;
    edges_a.reserve(a.size());

    for (size_t i = 0; i < a.size(); ++i) {
        int u = a[i];
        int v = a[(i + 1) % a.size()];
        if (u < v) 
            edges_a.emplace_back(u, v);
        else 
            edges_a.emplace_back(v, u);
    }

    // Sort to allow binary search (O(N log N))
    std::sort(edges_a.begin(), edges_a.end());

    size_t matches = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        int u = b[i];
        int v = b[(i + 1) % b.size()];
        // Create the normalized edge for b
        std::pair<int, int> edge = (u < v) ? std::make_pair(u, v) : std::make_pair(v, u);

        // Check if this edge exists in solution 'a'
        if (std::binary_search(edges_a.begin(), edges_a.end(), edge)) {
            matches++;
        }
    }
    return matches;
}

size_t similarity_nodes(const Solution& a, const Solution& b) {
    auto sorted_a = a;
    auto sorted_b = b;
    std::sort(sorted_a.begin(), sorted_a.end());
    std::sort(sorted_b.begin(), sorted_b.end());

    std::vector<int> intersection;
    std::set_intersection(sorted_a.begin(), sorted_a.end(), sorted_b.begin(), sorted_b.end(), std::back_inserter(intersection));
    return intersection.size();
}

void similarity_chart(
    const std::string& filename,
    const Solution& compare,
    const Solutions& solutions,
    const std::vector<int>& scores,
    SimilarityMeasure measure,
    std::function<bool(size_t)> skip = nullptr
) {
    std::ofstream similarity_csv(filename);
    similarity_csv << "objective_function,similarity" << std::endl;

    for (size_t i = 0; i < solutions.size(); i++) {
        if (skip && skip(i)) {
            continue;
        }
        
        similarity_csv << scores[i] << "," << measure(solutions[i], compare) << std::endl;
    }
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

    auto rng = std::mt19937{156053 + 156042};

    int solution_length = points.size() / 2; // - 1
    std::cout << std::endl;

    constexpr size_t local_optima_count = 1000u;
    Solutions local_optima;
    local_optima.reserve(local_optima_count);
    std::vector<int> scores;
    scores.reserve(local_optima_count);
    
    size_t best_1000_idx;
    {
        int best_score = std::numeric_limits<int>::max();
        for (size_t i = 0u; i < local_optima_count; ++i) {
            local_optima.emplace_back(local_search_steepest_edges(generate_random_solution(solution_length, points.size(), rng), distance_mat, node_costs));
            scores.emplace_back(calculate_objective_function(distance_mat, local_optima.back(), node_costs));
            if (scores.back() < best_score) {
                best_score = scores.back();
                best_1000_idx = i;
            }
            std::cout << "\r" << ProgressBar(i, local_optima_count) << std::flush;
        }
    }
    std::cout << std::endl;

    Solution best_overall;
    {
        Solutions solutions;
        auto avg_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double, std::milli>(3000));
        size_t max_runs = 20;

        constexpr int perturbation_strength = 21;
        std::cout << "Running ILS (ps=" << perturbation_strength << ")" << std::flush;
        for (size_t i = 0; i < max_runs; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            const auto result{ iterated_local_search(avg_ns, distance_mat, node_costs, solution_length, points.size(), rng, perturbation_strength) };
            solutions.emplace_back(std::get<0>(result));
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "\rRunning ILS (ps=" << perturbation_strength << ") " << ProgressBar(i, max_runs, max_runs) << " " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() * (max_runs - i - 1) << "s left     " << std::flush;
        }
        std::cout << std::endl;
        calculate_statistics(points, distance_mat, node_costs, solutions, instance, "ils", "ils");
        
        int best_score = std::numeric_limits<int>::max();
        for (const auto& solution : solutions) {
            int score = calculate_objective_function(distance_mat, solution, node_costs);
            if (score < best_score) {
                best_score = score;
                best_overall = solution;
            }
        }
        solutions.clear();
        std::cout << std::endl;
    }

    std::vector<std::pair<std::string, SimilarityMeasure>> measures = {
        {"nodes", similarity_nodes},
        {"edges", similarity_edges}
    };

    for (const auto& [name, measure] : measures) {
        // 1. Similarity to best of 1000
        similarity_chart(instance + "_" + name + "_best1000.csv", local_optima[best_1000_idx], local_optima, scores, measure, [=](size_t i) {
            return i == best_1000_idx;
        });

        // 2. Similarity to best overall (external)
        similarity_chart(instance + "_" + name + "_bestExternal.csv", best_overall, local_optima, scores, measure);

        // 3. Average similarity to others
        {
            std::ofstream avg_sim_csv(instance + "_" + name + "_avg.csv");
            avg_sim_csv << "objective_function,average_similarity" << std::endl;

            for (size_t i = 0; i < local_optima.size(); ++i) {
                double sum_similarity = 0.0;
                for (size_t j = 0; j < local_optima.size(); ++j) {
                    if (i == j) continue;
                    sum_similarity += measure(local_optima[i], local_optima[j]);
                }
                double avg_similarity = sum_similarity / (local_optima.size() - 1);
                avg_sim_csv << scores[i] << "," << avg_similarity << std::endl;
            }
        }
    }

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
