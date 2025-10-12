import kotlin.math.roundToInt
import kotlin.math.sqrt

data class Node(val index: Int, val x: Int, val y: Int, val cost: Int) {
    fun distanceTo(other: Node): Int {
        val dx = (this.x - other.x).toDouble()
        val dy = (this.y - other.y).toDouble()
        return sqrt(dx * dx + dy * dy).roundToInt()
    }
}