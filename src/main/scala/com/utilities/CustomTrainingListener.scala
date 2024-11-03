package com.utilities

import org.deeplearning4j.optimize.listeners.ScoreIterationListener

class CustomTrainingListener extends ScoreIterationListener(10) {
  private var currentScore: Double = 0.0

  override def iterationDone(model: org.deeplearning4j.nn.api.Model, iteration: Int, epochNum: Int): Unit = {
    super.iterationDone(model, iteration, epochNum)
    currentScore = model.score()
  }

  def getLastScore: Double = currentScore
}