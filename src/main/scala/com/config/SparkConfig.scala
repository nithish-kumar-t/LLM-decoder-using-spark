package com.config

import org.apache.spark.SparkConf
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray

object SparkConfig {
  def getSparkConf(): SparkConf = {
    new SparkConf()
      .setAppName("DistributedLanguageModel")
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
