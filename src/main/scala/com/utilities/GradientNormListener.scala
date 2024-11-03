package com.utilities

import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.optimize.api.IterationListener
import org.nd4j.linalg.api.ndarray.INDArray

class GradientNormListener(logFrequency: Int) extends IterationListener {
  private var iteration = 0

  //  override def invoked(): Boolean = true

  //  override def invoke(): Unit = {}

  override def iterationDone(model: Model, iteration: Int, epoch: Int): Unit = {
    if (iteration % logFrequency == 0) {
      // Get the gradients
      val gradients: INDArray = model.gradient().gradient()

      val gradientMean = gradients.meanNumber().doubleValue()
      val gradientMax = gradients.maxNumber().doubleValue()
      val gradientMin = gradients.minNumber().doubleValue()
      println(s"Iteration $iteration: Gradient Mean = $gradientMean, Max = $gradientMax, Min = $gradientMin")
    }
  }
}