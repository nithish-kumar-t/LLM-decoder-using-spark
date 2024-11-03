package com.traingDecoder

import com.config.{ConfigLoader, SparkConfig}
import com.utilities.{Environment, FileIO}
import org.apache.spark.SparkContext
import org.slf4j.LoggerFactory

import java.io.{BufferedWriter, FileWriter}
import java.nio.file.{Files, Paths}
import java.time.Instant

/**
 * TextGenerationInLLM is the main entry point for training and generating text with a language model.
 * It configures and runs a Spark-based training process, logs training metrics, and generates text samples.
 *
 *
 * Usage:
 * Run this object to train the model on the input dataset (specified in the configuration) and to generate sample text.
 * The configuration values, such as the number of epochs and word count to generate, are loaded from the application's
 * configuration file.
 */
object TextGenerationInLLM {
  private val logger = LoggerFactory.getLogger(getClass)



  /**
   * This Main method is the entry point of the whole program, It requires 2 arguments,
   * 1 -> Environment which the application is running.
   * 2 -> The input seed text file, sentences are generated based on the seed.
   */
  def main(args: Array[String]): Unit = {
    if (args.length <2) {
      logger.error("Missing params, Missing Input seed text file path")
      return
    }

    val env: String = args(0).split("=")(1)
    val seedFilePath: String = args(1)

    if (!Environment.values.exists(value => env.equals(value.toString))) {
      logger.error("Environment is invalid")
      return
    }


    if (!Files.exists(Paths.get(seedFilePath))) {
      logger.error("Input Seed text File path is invalid")
      return
    }

    // Configure Spark with appropriate resources
    val model = new SentenceGeneration()
    val sc = new SparkContext(SparkConfig.getSparkConf())
    val epochs = ConfigLoader.getConfig("epochs").toInt
    val numSentenceGen = ConfigLoader.getConfig("wordCountToGenerate").toInt

    val inputFilePath = ConfigLoader.getConfig(s"$env.inputPath")
    val outPutFolder = ConfigLoader.getConfig(s"$env.outputPath") + "/OUT-"+ Instant.now().getEpochSecond.toString
    FileIO.createFolder(outPutFolder)

    val metricsWriter = new BufferedWriter(new FileWriter(outPutFolder+"/training_metrics.csv"))
    //Format of metrics for CSV file
    metricsWriter.write("Epoch,\tLearningRate,\tLoss,\tAccuracy,\tBatchesProcessed,\tPredictionsMade,\tEpochDuration,\tNumber of partitions,\tNumber Of Lines, \tMemoryUsed\n")

    try {
      // Setting Spark logging Level to INFO, to get logs for each iteration.
      sc.setLogLevel("INFO")

      //We are caching the training data, which results in vast performance improvement, as very few calls to S3.
      val textRDD = sc.textFile(inputFilePath)
        .map(_.trim)
        .filter(_.nonEmpty)
        .cache()

      // Spark Partition Info
      logger.info(s"Number of partitions: ${textRDD.getNumPartitions}")
      logger.info(s"Total number of lines: ${textRDD.count()}")

      val trainedModel = model.train(sc, textRDD, metricsWriter, epochs)

      // Fitting the model with the input file.
      val tokenizer = new Tokenizer()
      val texts = textRDD.collect()
      tokenizer.fit(texts)

      // Generating new Sentences using initial Seed.
      val seedText: String = FileIO.readFileToStringNIO(seedFilePath)

      val generatedText = model.generateText(trainedModel, tokenizer, seedText, numSentenceGen)
      val cleanedText = generatedText.replaceAll("\\s+", " ")

      // Writing generated text into the file

      FileIO.writeStringToFileNIO(outPutFolder+"/generated-data.txt", s"Generated text:$seedText $cleanedText")

      logger.info(s"Cleaned text: $cleanedText")
      logger.info(s"Generated text: $generatedText")

    } finally {
      metricsWriter.close()
      sc.stop()
    }
  }
}
