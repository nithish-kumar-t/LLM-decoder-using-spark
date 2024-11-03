package com.utilities

/**
 * `Environment` is an enumeration that defines the different runtime environments for the application.
 * It provides a standardized way to specify whether the application is running locally, in a test
 * environment, or in a cloud environment. This enumeration can be used to configure environment-specific
 * paths, settings, and resources.
 *
 * Usage:
 * - Access these values as `Environment.local`, `Environment.test` or `Environment.cloud`.
 * - Useful for conditionally applying configurations based on the runtime environment.
 * - This value is provided to the application as part of the runtime configuration.
 */
object Environment extends Enumeration {
  type Environment = Value
  val local: Value = Value("local")
  val test: Value = Value("test")
  val cloud: Value = Value("cloud")
}