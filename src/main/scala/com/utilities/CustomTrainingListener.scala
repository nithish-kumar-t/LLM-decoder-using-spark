package com.utilities

import org.deeplearning4j.optimize.listeners.ScoreIterationListener


/**
 * CustomTrainingListener extends the default `ScoreIterationListener` to monitor the model's score (loss) at regular intervals.
 * It stores the latest score to provide insights into model performance after each iteration.
 * This listener is useful for tracking and retrieving the model's score in real-time during training.
 */
class CustomTrainingListener extends ScoreIterationListener(10) {
  //Using a var to keep track of model performance, this object needs to mutate.
  private var currentScore: Double = 0.0

  /**
   * Called at each training iteration. This method records the model's score after each iteration.
   *
   * @param model The model being trained.
   * @param iteration The current training iteration.
   * @param epochNum The current epoch of training.
   */
  override def iterationDone(model: org.deeplearning4j.nn.api.Model, iteration: Int, epochNum: Int): Unit = {
    super.iterationDone(model, iteration, epochNum)
    currentScore = model.score()
  }

  def getLastScore: Double = currentScore
}