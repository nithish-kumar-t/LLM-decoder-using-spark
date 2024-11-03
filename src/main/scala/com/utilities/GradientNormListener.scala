package com.utilities

import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.optimize.api.IterationListener
import org.nd4j.linalg.api.ndarray.INDArray

/**
 * GradientNormListener is a custom listener for monitoring the norms of gradients during model training.
 * It logs the mean, maximum, and minimum values of the gradients every specified number of iterations,
 * providing insight into gradient behavior, which can be useful for debugging and ensuring model stability.
 *
 * @param logFrequency Frequency (in number of iterations) at which to log the gradient norms.
 */
class GradientNormListener(logFrequency: Int) extends IterationListener {

  /**
   * Logs the mean, maximum, and minimum of the model's gradients at every specified interval.
   *
   * @param model The neural network model being trained.
   * @param iteration The current iteration count in training.
   * @param epoch The current epoch count in training.
   */
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