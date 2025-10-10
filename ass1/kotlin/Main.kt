import java.io.File
import kotlin.math.sqrt

fun main() {
    val filename = "data/TSPA.csv"

    val data = DataReader.readData(filename)

    val targetNumberOfNodes = data.size / 2

    val distanceMatrix = data.map {
        node -> data.map {
            otherNode -> if (node != otherNode) node.distanceTo(otherNode) + otherNode.cost else Int.MAX_VALUE
        }
    }

    val bestSolutionRandom = getStatisticsAndBestSolution(::random, data, distanceMatrix, targetNumberOfNodes)
    exportSolutionToCSV(data, bestSolutionRandom.path)

    val bestSolutionNearestNeighbor = getStatisticsAndBestSolution(::nearestNeighbor, data, distanceMatrix, targetNumberOfNodes)
    exportSolutionToCSV(data, bestSolutionNearestNeighbor.path)

    val bestSolutionNearestNeighborInsertAnywhere = getStatisticsAndBestSolution(::nearestNeighborInsertAnywhere, data, distanceMatrix, targetNumberOfNodes)
    exportSolutionToCSV(data, bestSolutionNearestNeighborInsertAnywhere.path)
}

data class Solution(val path: List<Int>, val cost: Int) {}

fun random(startingNode: Int, distanceMatrix: List<List<Int>>, targetNumberOfNodes: Int): Solution {
    val path = distanceMatrix.indices.filter { it != startingNode }.shuffled().take(targetNumberOfNodes)
    return Solution(path, calculateObjectiveFunction(path, distanceMatrix))
}

fun nearestNeighbor(startingNode: Int, distanceMatrix: List<List<Int>>, targetNumberOfNodes: Int): Solution {
    val unvisited = distanceMatrix.indices.toMutableSet()
    val path = mutableListOf<Int>()
    var currentNode = startingNode
    path.add(currentNode)
    unvisited.remove(currentNode)

    while (path.size < targetNumberOfNodes) {
        val nextNode = unvisited.minByOrNull { distanceMatrix[currentNode][it] }!!
        path.add(nextNode)
        unvisited.remove(nextNode)
        currentNode = nextNode
    }

    return Solution(path, calculateObjectiveFunction(path, distanceMatrix))
}

fun nearestNeighborInsertAnywhere(startNode: Int, distanceMatrix: List<List<Int>>, targetSize: Int): Solution {
    val unvisited = distanceMatrix.indices.toMutableSet()
    val path = mutableListOf(startNode)
    unvisited.remove(startNode)

    while (path.size < targetSize) {
        var bestNode = -1
        var bestPosition = -1
        var bestDelta = Int.MAX_VALUE

        for (candidate in unvisited) {
            for (i in 0..path.size) {
                val before = path.getOrNull(i - 1) ?: path.last()
                val after = path.getOrNull(i % path.size) ?: path.first()
                val delta = distanceMatrix[before][candidate] + distanceMatrix[candidate][after] -
                        distanceMatrix[before][after]

                if (delta < bestDelta) {
                    bestDelta = delta
                    bestNode = candidate
                    bestPosition = i
                }
            }
        }

        path.add(bestPosition, bestNode)
        unvisited.remove(bestNode)
    }

    return Solution(path, calculateObjectiveFunction(path, distanceMatrix))
}

fun exportSolutionToCSV(
    data: List<Node>,
    solution: List<Int>,
    filename: String = "solution.csv"
) {
    File(filename).printWriter().use { out ->
        out.println("index,x,y,cost,selected")
        for (node in data) {
            val selected = if (solution.contains(node.index)) 1 else 0
            out.println("${node.index},${node.x},${node.y},${node.cost},$selected")
        }
    }

    File("path.csv").printWriter().use { out ->
        for (i in solution.indices) {
            val from = data[solution[i]]
            val to = data[solution[(i + 1) % solution.size]]
            out.println("${from.x},${from.y}")
            out.println("${to.x},${to.y}")
            out.println("") // Blank line to separate lines in gnuplot
        }
    }
}

fun calculateObjectiveFunction(path: List<Int>, distanceMatrix: List<List<Int>>): Int {
    var totalCost = 0
    for (i in path.indices) {
        val from = path[i]
        val to = path[(i + 1) % path.size]
        totalCost += distanceMatrix[from][to]
    }
    return totalCost;
}

fun getStatisticsAndBestSolution(method: (Int, List<List<Int>>, Int) -> Solution, data: List<Node>, distanceMatrix: List<List<Int>>, targetNumberOfNodes: Int): Solution {
    val solutions = mutableListOf<Solution>()
    for (i in data) {
        val solution = method(i.index, distanceMatrix, targetNumberOfNodes)
        solutions.add(solution)
    }

    val bestSolution = solutions.minByOrNull { it.cost }!!
    val averageCost = solutions.map { it.cost }.average()
    val worstSolutionCost = solutions.maxByOrNull { it.cost }!!.cost

    println("Best solution cost: ${bestSolution.cost}, Path: ${bestSolution.path}")
    println("Average cost: $averageCost")
    println("Worst cost: $worstSolutionCost")

    return bestSolution
}
