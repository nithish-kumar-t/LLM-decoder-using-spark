import SlidingWindowSpark.SlidingWindowUtils.WindowedData
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._

object SlidingWindowSpark {
  def main(args: Array[String]): Unit = {
    // Set up Spark configuration and context
    val conf = new SparkConf().setAppName("Sliding Window Dataset").setMaster("local")
    val sc = new SparkContext(conf)

    // Example input data (could be sentences, tokens, etc.)
    val sentences = Array(
      "The quick brown fox jumps over the lazy dog",
      "This is another sentence for testing sliding windows"
    )

    // Parallelize the input data (convert array to an RDD)
    val sentenceRDD: RDD[String] = sc.parallelize(sentences)

    // Apply the sliding window logic to create the dataset
    val slidingWindowDataset: RDD[WindowedData] = sentenceRDD.flatMap { sentence =>
      SlidingWindowUtils.createSlidingWindows(sentence, 4).asScala
    }


    slidingWindowDataset.map(window =>
      (window.getInput.mkString(", ")+ "-----", window.getTarget)).saveAsTextFile("src/main/resources/output/Sliding-Window-Spark-2")

    // Stop the Spark context
    sc.stop()
  }

  object SlidingWindowUtils {
    // Define WindowedData class with Serializable
    class WindowedData(private val input: Array[String], private val target: String) extends Serializable {
      def getInput: Array[String] = input
      def getTarget: String = target
    }

    // Create sliding windows for a given sentence
    def createSlidingWindows(sentence: String, windowSize: Int): java.util.List[WindowedData] = {
      val tokens = sentence.split(" ")
      val windowedDataList = new java.util.ArrayList[WindowedData]()

      // Create sliding windows
      for (i <- 0 until tokens.length - windowSize) {
        val inputWindow = tokens.slice(i, i + windowSize) // Use slice for cleaner code
        val target = tokens(i + windowSize)
        windowedDataList.add(new WindowedData(inputWindow, target))
      }

      windowedDataList
    }
  }
}
