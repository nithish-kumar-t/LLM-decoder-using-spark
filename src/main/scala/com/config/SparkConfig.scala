package com.config

import org.apache.spark.SparkConf
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

/**
 * SparkConfig is a utility object that provides a predefined Spark configuration for a distributed language model application.
 *
 * Key parameters in this configuration:
 * - `setAppName("LLM-Decoder-for-Distributed-processing")`: Sets the application name, useful for identifying it in Spark UIs and logs.
 * - `setMaster("local[*]")`: Specifies the master URL, allowing the application to run locally with as many threads as available cores.
 * - `spark.executor.memory` and `spark.driver.memory`: Defines memory allocation for the Spark executors and driver, respectively.
 * - `spark.serializer`: Sets Kryo as the serialization library, offering faster and more compact serialization compared to Java serialization.
 * - `spark.kryoserializer.buffer.max` and `spark.kryoserializer.buffer`: Configures the maximum and initial buffer sizes for Kryo serialization.
 *
 * Additionally, this configuration registers Kryo classes that are commonly used in deep learning workflows:
 * - `MultiLayerNetwork`: Represents a neural network model from the DL4J library.
 * - `INDArray`: Represents n-dimensional arrays used by DL4J for tensor operations.
 * - `Array[Byte]` and `DataBuffer`: Supports serialization of byte arrays and data buffers needed for deep learning computations.
 *
 * This configuration is typically used in distributed deep learning tasks that require high memory
 * and efficient serialization/deserialization of neural network models.
 */
object SparkConfig {
  def getSparkConf(): SparkConf = {
    new SparkConf()
      .setAppName("LLM-Decoder-for-Distributed-processing")
      .setMaster("local[*]")
      .set("spark.executor.memory", "4g")
      .set("spark.driver.memory", "4g")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.kryoserializer.buffer.max", "512m")
      .set("spark.kryoserializer.buffer", "256m")
      .registerKryoClasses(Array(
        classOf[MultiLayerNetwork],
        classOf[INDArray],
        classOf[Array[Byte]],
        classOf[org.nd4j.linalg.api.buffer.DataBuffer]
      ))
  }
}
