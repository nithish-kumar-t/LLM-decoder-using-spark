package com.traingDecoder

import com.config.{ConfigLoader, SparkConfig}
import org.apache.spark.SparkContext
import org.slf4j.LoggerFactory

import java.io.{BufferedWriter, FileWriter}

object TextGenerationInLLM {
  private val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    // Configure Spark with appropriate resources
    val model = new SentenceGeneration()
    val sc = new SparkContext(SparkConfig.getSparkConf())
    val epochs = ConfigLoader.getConfig("epochs").toInt
    val numSentenceGen = ConfigLoader.getConfig("wordCountToGenerate").toInt

    val metricsFilePath = "src/main/resources/training_metrics-1.csv"

    val metricsWriter = new BufferedWriter(new FileWriter(metricsFilePath))
    //Format of metrics for CSV file
    metricsWriter.write("Epoch,\tLearningRate,\tLoss,\tAccuracy,\tBatchesProcessed,\tPredictionsMade,\tEpochDuration,\tNumber of partitions,\tNumber Of Lines, \tMemoryUsed\n")

    try {
      // Enable logging
      sc.setLogLevel("INFO")

      val filePath = "src/main/resources/input.txt"
      val textRDD = sc.textFile(filePath)
        .map(_.trim)
        .filter(_.nonEmpty)
        .cache()

      // Print initial statistics
      logger.info(s"Number of partitions: ${textRDD.getNumPartitions}")
      logger.info(s"Total number of lines: ${textRDD.count()}")

      val trainedModel = model.train(sc, textRDD, metricsWriter, epochs)

      // Generate sample text
      val tokenizer = new Tokenizer()
      val texts = textRDD.collect()
      tokenizer.fit(texts)

      val generatedText = model.generateText(trainedModel, tokenizer, "scientist", numSentenceGen)
      val cleanedText = generatedText.replaceAll("\\s+", " ")
      logger.info(s"Cleaned text: $cleanedText")
      logger.info(s"Generated text: $generatedText")

    } finally {
      metricsWriter.close()
      sc.stop()
    }
  }
}
