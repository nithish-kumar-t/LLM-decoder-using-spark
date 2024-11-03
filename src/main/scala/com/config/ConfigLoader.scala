package com.config

import com.typesafe.config.{Config, ConfigFactory}

/**
 * ConfigLoader is a utility object that loads application configurations using Typesafe Config.
 * It provides methods to retrieve the entire configuration or specific values by key.
 */
object ConfigLoader {
  val config: Config = ConfigFactory.load()
  def loadConfig(): Config = {
    config
  }

  def getConfig(key : String) : String = {
    config.getString(key)
  }
}
