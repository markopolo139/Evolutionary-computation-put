object DataReader {
    fun readData(fileName: String): List<Node> {
        val nodes = mutableListOf<Node>()
        val file = java.io.File(fileName)
        var index = 0;
        file.forEachLine { line ->
            val parts = line.split(";").map { it.trim() }
            if (parts.size == 3) {
                val (x, y, cost) = parts.map { it.toInt() }
                nodes.add(Node(index++, x, y, cost))
            }
            else {
                println("Invalid number of parts in line: $line")
            }
        }
        return nodes
    }
}