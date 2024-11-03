package com.traingDecoder

import com.config.{ConfigLoader, SparkConfig}
import com.utilities.{Environment, FileIO}
import org.apache.spark.SparkContext
import org.slf4j.LoggerFactory

import java.io.{BufferedWriter, FileWriter}
import java.nio.file.{Files, Paths}
import java.time.Instant
import scala.collection.mutable.ListBuffer

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

    //if it is a s3 path, then don't do this check, else do this check
    if (!seedFilePath.startsWith("s3") && !Files.exists(Paths.get(seedFilePath))) {
      logger.error("Input Seed text File path is invalid")
      return
    }

    // Configure Spark with appropriate resources
    val model = new SentenceGeneration()
    val sc = new SparkContext(SparkConfig.getSparkConf())
    val epochs = ConfigLoader.getConfig(s"$env.epochs").toInt
    val numSentenceGen = ConfigLoader.getConfig("wordCountToGenerate").toInt

    // Set up the TrainingMaster configuration
//    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
//      .batchSizePerWorker(32)         // Batch size on each Spark worker
//      .averagingFrequency(5)          // Frequency of parameter averaging
//      .workerPrefetchNumBatches(2)    // Number of batches to prefetch per worker
//      .build()

    val inputFilePath = ConfigLoader.getConfig(s"$env.inputPath")
    val outPutFolder = ConfigLoader.getConfig(s"$env.outputPath") + "/OUT-"+ Instant.now().getEpochSecond.toString
    if (outPutFolder.startsWith("s3")) {
      createFolders3(sc, outPutFolder)
    } else {
      FileIO.createFolder(outPutFolder)
    }

    val metricsBuffer = new StringBuilder()
    val metricsPath = outPutFolder+"/training_metrics.csv"
    //Format of metrics for CSV file
    metricsBuffer.append(ConfigLoader.getConfig("csv-header"))

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

      val trainedModel = model.train(sc, textRDD, metricsBuffer, epochs)

      // Fitting the model with the input file.
      val tokenizer = new Tokenizer()
      val texts = textRDD.collect()
      tokenizer.fit(texts)

      // Generating new Sentences using initial Seed.
      // Read the seed text from S3 using Spark if path starts with s3
      val seedText: String = if (seedFilePath.startsWith("s3")) {
        sc.textFile(seedFilePath).collect().mkString(" ")
      } else {
        FileIO.readFileToStringNIO(seedFilePath)
      }

      val generatedText = model.generateText(trainedModel, tokenizer, seedText, numSentenceGen)
      val cleanedText = generatedText.replaceAll("\\s+", " ")
      // Writing generated text into the file

      val outputPath : String = outPutFolder+"/generated-data.txt"
      val data : String = s"Generated text:$seedText $cleanedText"
      logger.info("outputPath::::" + outputPath)
      logger.info("outputData::::\t" + data)

      if (env.equals("cloud")) {
        //write output to s3 (for env cloud)
        writeIntoS3(sc, metricsPath, metricsBuffer.mkString)
        writeIntoS3(sc, outputPath, data)
      } else {
        // Flushing into files for (Local, Test)
        logger.debug(metricsBuffer.mkString(" "))
        FileIO.writeStringToFileNIO(outputPath, data)
        val metricsWriter = new BufferedWriter(new FileWriter(metricsPath))
        metricsWriter.write(metricsBuffer.mkString)
        metricsWriter.close()
      }

      logger.info(s"Cleaned text: $cleanedText")
      logger.info(s"Generated text: $generatedText")

    } finally {
//      metricsWriter.close()
      sc.stop()
    }
  }

  private def createFolders3(sc : SparkContext, outPutFolder: String) : Unit = {
    val rdd = sc.emptyRDD[String]
    rdd.saveAsTextFile(outPutFolder)
  }

  private def writeIntoS3(sc : SparkContext, outPutPath: String, data : String) : Unit = {
    val genRdd = sc.parallelize(Seq(data))
    genRdd.saveAsTextFile(outPutPath)
  }

}
