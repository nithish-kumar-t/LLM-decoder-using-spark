package com.config

import com.utilities.{CustomTrainingListener, GradientNormListener}
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.EvaluativeListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.schedule.{ExponentialSchedule, ScheduleType}


/**
 * MultiNetworkModel provides functionality to build and initialize a neural network model
 * for language tasks using a multi-layer configuration.
 * It is configured to use custom settings
 * for layer sizes, activation functions, and dropout to optimize training performance and accuracy.
 *
 * Configurable parameters (such as `vocabularySize`, `embeddingSize`, `windowSize`, and `batchSize`)
 * are retrieved from a configuration loader for flexibility.
 */
object MultiNetworkModel {
  val vocabularySize: Int = ConfigLoader.getConfig("common.vocabularySize").toInt
  val embeddingSize: Int = ConfigLoader.getConfig("common.embeddingSize").toInt
  val windowSize: Int = ConfigLoader.getConfig("common.windowSize").toInt
  val batchSize: Int = ConfigLoader.getConfig("common.batchSize").toInt


  /**
   * Builds and initializes a neural network model for training.
   *
   * @return Initialized MultiLayerNetwork model with configured layers.
   */
  def buildModel(dataSetIterator: DataSetIterator): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .seed(42)
      .updater(new Adam(new ExponentialSchedule(ScheduleType.EPOCH, 0.005, 0.9)))
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new DenseLayer.Builder()
        .nIn(embeddingSize * windowSize)
        .nOut(128)
        .activation(Activation.RELU)
        .dropOut(0.2)
        .build())
      .layer(1, new DenseLayer.Builder()
        .nIn(512)
        .nOut(128)
        .activation(Activation.RELU)
        .dropOut(0.2)
        .build())
      .layer(2, new OutputLayer.Builder()
        .nIn(128)
        .nOut(vocabularySize)
        .activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .build())
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()

    // Add custom listener for monitoring
    val listener = new CustomTrainingListener
    model.setListeners(listener, new GradientNormListener(10), new EvaluativeListener(dataSetIterator, 1))
    model
  }
}

