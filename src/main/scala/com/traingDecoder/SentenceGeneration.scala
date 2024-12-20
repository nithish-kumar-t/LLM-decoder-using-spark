package com.traingDecoder

import org.apache.commons.io.output.ByteArrayOutputStream
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.ops.transforms.Transforms

import java.io._
import scala.collection.JavaConverters._
import scala.collection.mutable.{ArrayBuffer}
import scala.util.Random
import com.config.MultiNetworkModel
import org.slf4j.LoggerFactory
import com.config.MultiNetworkModel._

/**
 * SentenceGeneration is a class for training a language model using a neural network
 * in a distributed Spark environment. It includes methods for model serialization,
 * creating training batches with sliding windows, attention mechanisms, and model training.
 * The class also supports generating text based on a seed phrase using temperature sampling.
 *
 * This class uses Spark RDDs for distributed processing, enabling scalable training on large datasets.
 */
class SentenceGeneration extends Serializable {
  private val logger = LoggerFactory.getLogger(getClass)


  /**
   * Serializes a given MultiLayerNetwork model to an array of bytes.
   *
   * @param model The neural network model to serialize.
   * @return Array of bytes representing the serialized model.
   */
  private def serializeModel(model: MultiLayerNetwork): Array[Byte] = {
    val baos = new ByteArrayOutputStream()
    try {
      val oos = new ObjectOutputStream(baos)
      oos.writeObject(model.params())
      oos.writeObject(model.getLayerWiseConfigurations)
      oos.close()
      baos.toByteArray
    } finally {
      baos.close()
    }
  }

  /**
   * Deserializes a MultiLayerNetwork model from an array of bytes.
   *
   * @param bytes Byte array containing the serialized model data.
   * @return Deserialized MultiLayerNetwork object.
   */
  private def deserializeModel(bytes: Array[Byte]): MultiLayerNetwork = {
    val bais = new ByteArrayInputStream(bytes)
    try {
      val ois = new ObjectInputStream(bais)
      val params = ois.readObject().asInstanceOf[INDArray]
      val conf = ois.readObject().asInstanceOf[org.deeplearning4j.nn.conf.MultiLayerConfiguration]
      val model = new MultiLayerNetwork(conf)
      model.init()
      model.setParams(params)
      model
    } finally {
      bais.close()
    }
  }


  /**
   * Creates sliding windows from tokenized sequences to generate training samples.
   *
   * @param tokens Sequence of integer tokens representing words.
   * @return Sequence of pairs where each pair contains a context window and the target token.
   */
  private def createSlidingWindows(tokens: Seq[Int]): Seq[(Seq[Int], Int)] = {
    tokens.sliding(windowSize + 1).map { window =>
      (window.init, window.last)
    }.toSeq
  }

  /**
   * Applies a self-attention mechanism on the input matrix.
   *
   * @param input Input INDArray with shape (batchSize, sequenceLength, embedSize).
   * @return INDArray with the attention-applied values.
   */
  private def selfAttention(input: INDArray): INDArray = {
    val Array(batchSize, sequenceLength, embedSize) = input.shape()

    // Create query, key, and value matrices for each batch independently
    val query = Nd4j.createUninitialized(batchSize, sequenceLength, embedSize)
    val key = Nd4j.createUninitialized(batchSize, sequenceLength, embedSize)
    val value = Nd4j.createUninitialized(batchSize, sequenceLength, embedSize)

    // Ensure query, key, and value are initialized properly
    if (query.isEmpty || key.isEmpty || value.isEmpty) {
      return Nd4j.empty()
    }

    // Compute the dot product between queries and keys
    val scores = query
      .tensorAlongDimension(0, 1, 2)
      .mmul(key.tensorAlongDimension(0, 1, 2).transpose())
      .div(math.sqrt(embedSize))

    // Apply softmax along the last dimension to get attention weights
    val attentionWeights = Transforms.softmax(scores)

    // Multiply the weights with the values
    val attendedOutput = attentionWeights
      .tensorAlongDimension(0, 1, 2)
      .mmul(value.tensorAlongDimension(0, 1, 2))

    attendedOutput.reshape(batchSize, sequenceLength, embedSize)
  }

  /**
   * Creates an embedding matrix with positional encodings for a given sequence.
   *
   * @param sequence Sequence of integer tokens representing words.
   * @return INDArray representing the embedding matrix with positional encodings.
   */
  def createEmbeddingMatrix(sequence: Seq[Int]): INDArray = {
    val embedding = Nd4j.zeros(1, embeddingSize, sequence.length)

    // Create word embeddings
    sequence.zipWithIndex.foreach { case (token, pos) =>
      val tokenEmbedding = Nd4j.randn(1, embeddingSize).mul(0.1)
      embedding.putSlice(pos, tokenEmbedding)
    }

    sequence.indices.foreach { pos =>
      Iterator.range(0, embeddingSize).foreach { i =>
        val angle = pos / math.pow(10000, (2 * i).toFloat / embeddingSize)
        val updatedValue = if (i % 2 == 0) {
          embedding.getDouble(0, i, pos) + math.sin(angle)
        } else {
          embedding.getDouble(0, i, pos) + math.cos(angle)
        }
        embedding.putScalar(Array(0, i, pos), updatedValue)
      }
    }
    embedding
  }

  /**
   * Creates a validation DataSetIterator for model evaluation from an RDD of validation data.
   *
   * @param validationDataRDD RDD of text strings for validation.
   * @param tokenizer Tokenizer object for encoding text into token sequences.
   * @return DataSetIterator for validation data.
   */
  def createValidationDataSetIterator(validationDataRDD: RDD[String], tokenizer: Tokenizer): DataSetIterator = {
    // Process the validation data to create features and labels
    val validationData = validationDataRDD.flatMap { text =>
      val tokens = tokenizer.encode(text)
      createSlidingWindows(tokens).map { case (inputSeq, label) =>
        val inputArray = Nd4j.zeros(1, embeddingSize * windowSize)
        val labelArray = Nd4j.zeros(1, vocabularySize)

        // Convert input sequence and label to ND4J arrays
        val embedding = createEmbeddingMatrix(inputSeq)
        val attentionOutput = selfAttention(embedding)
        if (!attentionOutput.isEmpty) {
          val flattenedAttention = attentionOutput.reshape(1, embeddingSize * windowSize)
          inputArray.putRow(0, flattenedAttention)
          labelArray.putScalar(Array(0, label), 1.0)

          new DataSet(inputArray, labelArray)
        }
        new DataSet()
      }
    }.collect().toList.asJava

    // Create a ListDataSetIterator with a batch size of 1 (or adjust as needed)
    new ListDataSetIterator(validationData, batchSize)
  }

  /**
   * Trains the language model using distributed training with Spark and saves metrics.
   *
   * @param sc SparkContext to be used for distributed training.
   * @param textRDD RDD containing the text data for training.
   * @param metricsBuffer ListBuffer for logging training metrics.
   * @param trainingMaster is used by sparkDl4j for parallel processing
   * @param epochs Number of training epochs to run.
   * @return The trained MultiLayerNetwork model.
   */
  // Model training
  def train(sc: SparkContext, textRDD: RDD[String], metricsBuffer: StringBuilder, epochs: Int): MultiLayerNetwork = {
    val tokenizer = new Tokenizer()
    val allTexts = textRDD.collect()
    tokenizer.fit(allTexts)
    val broadcastTokenizer = sc.broadcast(tokenizer)

    // Split textRDD into training and validation sets
    val Array(trainingDataRDD, validationDataRDD) = textRDD.randomSplit(Array(0.8, 0.2))

    // Generating validation DataSetIterator
    val validationDataSetIterator = createValidationDataSetIterator(validationDataRDD, tokenizer)

    // Initial model and serialization
    val initialModel = MultiNetworkModel.buildModel(validationDataSetIterator)
    val initialModelBytes = serializeModel(initialModel)

    // Accumulators for metrics
    val batchProcessedAcc = sc.longAccumulator("batchesProcessed")
    val totalLossAcc = sc.doubleAccumulator("totalLoss")
    val correctPredictionsAcc = sc.longAccumulator("correctPredictions")
    val totalPredictionsAcc = sc.longAccumulator("totalPredictions")

    // Run epochs using foldLeft to update model and metrics
    Iterator.range(0 , epochs).foldLeft((initialModelBytes, initialModel)) { case ((modelBytes, model), epoch) =>
      val broadcastModel = sc.broadcast(modelBytes)
      val epochStartTime = System.currentTimeMillis()
      logger.info(s"Starting epoch $epoch")

      // Retrieve learning rate from optimizer
      val learningRate = model.getLayerWiseConfigurations.getConf(0).getLayer
        .asInstanceOf[org.deeplearning4j.nn.conf.layers.BaseLayer]
        .getIUpdater.asInstanceOf[Adam].getLearningRate(epoch, epochs)
      logger.info(s"Effective learning rate for epoch $epoch: $learningRate")

      // Reset accumulators
      batchProcessedAcc.reset()
      totalLossAcc.reset()
      correctPredictionsAcc.reset()
      totalPredictionsAcc.reset()

      // Prepare RDD for training
      val samplesRDD = trainingDataRDD.flatMap { text =>
        val tokens = broadcastTokenizer.value.encode(text)
        createSlidingWindows(tokens)
      }.persist()

      // Process training in each partition and collect serialized model updates
      val updatedModels = samplesRDD.mapPartitions { partition =>
        val localModel = deserializeModel(broadcastModel.value)
        val batchBuffer = ArrayBuffer[(Seq[Int], Int)]()
        //These are very much necessary to keep track of model accuracy.
        var localLoss = 0.0
        var localCorrect = 0L
        var localTotal = 0L

        partition.foreach { sample =>
          batchBuffer += sample
          if (batchBuffer.size >= batchSize) {
            val (loss, correct, total) = processBatch(localModel, batchBuffer.toSeq)
            localLoss += loss
            localCorrect += correct
            localTotal += total
            batchBuffer.clear()
            batchProcessedAcc.add(1)
          }
        }

        if (batchBuffer.nonEmpty) {
          val (loss, correct, total) = processBatch(localModel, batchBuffer.toSeq)
          localLoss += loss
          localCorrect += correct
          localTotal += total
          batchProcessedAcc.add(1)
        }

        totalLossAcc.add(localLoss)
        correctPredictionsAcc.add(localCorrect)
        totalPredictionsAcc.add(localTotal)

        Iterator.single(serializeModel(localModel))
      }.collect()

      // Average models if multiple partitions
      val averagedModel = if (updatedModels.nonEmpty) {
        if (updatedModels.length > 1) {
          val models = updatedModels.map(deserializeModel)
          averageModels(models)
        } else {
          deserializeModel(updatedModels(0))
        }
      } else model

      // Serialize and broadcast updated model for the next epoch
      val newModelBytes = serializeModel(averagedModel)
      val epochDuration = System.currentTimeMillis() - epochStartTime
      val avgLoss = totalLossAcc.value / batchProcessedAcc.value
      val accuracy = if (totalPredictionsAcc.value > 0) {
        correctPredictionsAcc.value.toDouble / totalPredictionsAcc.value
      } else 0.0

      // Log metrics
      logger.info(f"""
                     |Epoch $epoch Statistics:
                     |Duration: ${epochDuration}ms
                     |Average Loss: $avgLoss%.4f
                     |Accuracy: ${accuracy * 100}%.2f%%
                     |Batches Processed: ${batchProcessedAcc.value}
                     |Predictions Made: ${totalPredictionsAcc.value}
                     |Memory Used: ${Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()}B
        """.stripMargin)

      val executorMemoryStatus = sc.getExecutorMemoryStatus.map { case (executor, (maxMemory, remainingMemory)) =>
        s"$executor: Max Memory = $maxMemory, Remaining Memory = $remainingMemory"
      }
      logger.info(s"Executor Memory Status:\n${executorMemoryStatus.mkString("\n")}")

      val metrics = f"$epoch, $learningRate%.6f, $avgLoss%.4f, ${accuracy * 100}%.2f, ${batchProcessedAcc.value}, ${totalPredictionsAcc.value}, $epochDuration, ${textRDD.getNumPartitions}, ${textRDD.count()}, ${executorMemoryStatus.mkString("\n")}\n"
      metricsBuffer.append(metrics)

      samplesRDD.unpersist()
      broadcastModel.destroy()
      (newModelBytes, averagedModel)
    }._2
  }


  /**
   * Generates text using a trained model based on a seed text and a specified length.
   * The generation process uses temperature sampling for creativity in word selection.
   *
   * @param model Trained MultiLayerNetwork model.
   * @param tokenizer Tokenizer object for encoding and decoding text.
   * @param seedText Initial text seed for generation.
   * @param length Number of words to generate.
   * @param temperature Sampling temperature for controlling randomness.
   * @return Generated text string.
   */
  def generateText(model: MultiLayerNetwork, tokenizer: Tokenizer, seedText: String, length: Int, temperature: Double = 0.7): String = {
    //we need a mutable variable to keep track of old values for each iteration.
    var currentSequence = tokenizer.encode(seedText).takeRight(windowSize)
    val generated = new ArrayBuffer[Int]()
    val rand = new Random()

    def sampleWithTemperature(logits: INDArray, temp: Double): Int = {
      val scaled = logits.div(temp)
      val expScaled = Transforms.exp(scaled)
      val probs = expScaled.div(expScaled.sum(1))

      // Convert to probabilities and sample
      val probArray = (0 until probs.columns()).map(i => probs.getDouble(i.toLong)).toArray

      // Sample using cumulative probabilities
      val cumSum = probArray.scanLeft(0.0)(_ + _).tail
      val sample = rand.nextDouble()
      cumSum.zipWithIndex.find(_._1 >= sample).map(_._2).getOrElse(0)
    }

    Iterator.range(0, length).foreach { _ => {
      val embedding = createEmbeddingMatrix(currentSequence)
      val attentionOutput = selfAttention(embedding)
      val flattenedAttention = attentionOutput.reshape(1, embeddingSize * windowSize)

      val output = model.output(flattenedAttention)
      val nextTokenIndex = sampleWithTemperature(output, temperature)

      generated += nextTokenIndex
      currentSequence = (currentSequence.tail :+ nextTokenIndex).takeRight(windowSize)
    }}

    tokenizer.decode(generated)
  }


  /**
   * Processes a batch of samples to train the model and calculate performance metrics.
   *
   * @param model MultiLayerNetwork model to be trained.
   * @param batch Sequence of input-target pairs.
   * @return Tuple containing loss, correct predictions count, and total samples.
   */
  private def processBatch(model: MultiLayerNetwork, batch: Seq[(Seq[Int], Int)]): (Double, Long, Long) = {
    val inputArray = Nd4j.zeros(batch.size, embeddingSize * windowSize)
    val labelsArray = Nd4j.zeros(batch.size, vocabularySize)

    batch.zipWithIndex.foreach { case ((sequence, label), idx) =>
      val embedding = createEmbeddingMatrix(sequence)
      val attentionOutput = selfAttention(embedding)
      if (!attentionOutput.isEmpty) {
        val flattenedAttention = attentionOutput.reshape(1, embeddingSize * windowSize)
        inputArray.putRow(idx, flattenedAttention)
        labelsArray.putScalar(Array(idx, label), 1.0)
      }
    }
    // Train on batch
    model.fit(inputArray, labelsArray)


    // Calculate metrics
    val output = model.output(inputArray)
    val predictions = Nd4j.argMax(output, 1)
    val labels = Nd4j.argMax(labelsArray, 1)
    // Calculate the number of correct predictions
    val correct = predictions.eq(labels).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
      .sumNumber().longValue()

    (model.score(), correct, batch.size)
  }

  /**
   * Averages parameters across multiple models to create a combined model.
   *
   * @param models Array of MultiLayerNetwork models to average.
   * @return A new MultiLayerNetwork model with averaged parameters.
   */
  private def averageModels(models: Array[MultiLayerNetwork]): MultiLayerNetwork = {
    val firstModel = models(0)
    if (models.length == 1) return firstModel

    val params = models.map(_.params())
    val avgParams = params.reduce((a, b) => a.add(b)).div(models.length)

    val result = new MultiLayerNetwork(firstModel.getLayerWiseConfigurations)
    result.init()
    result.setParams(avgParams)
    result
  }
}



