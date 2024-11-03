package com.utilities

import org.slf4j.LoggerFactory

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

/**
 * FileIO provides utility methods for basic file operations using Java NIO. It supports
 * reading content from a file as a string, writing a string to a file, and creating a directory
 * if it does not already exist. Logging is included for tracking file operations.
 */
object FileIO {
  private val logger = LoggerFactory.getLogger(getClass)

  /**
   * Reads the entire content of a file as a single string.
   *
   * @param filePath The path of the file to be read.
   * @return The content of the file as a string.
   */
  def readFileToStringNIO(filePath: String): String = {
    Files.readString(Paths.get(filePath))
  }

  /**
   * Writes a string to a file, overwriting any existing content.
   *
   * @param filePath The path of the file where the content will be written.
   * @param content The content to write to the file.
   */
  def writeStringToFileNIO(filePath: String, content: String): Unit = {
    Files.write(Paths.get(filePath), content.getBytes(StandardCharsets.UTF_8))
  }

  /**
   * Creates a directory if it does not already exist.
   *
   * @param folderPath The path of the directory to create.
   * Logs a message indicating whether the directory was created or already existed.
   */
  def createFolder(folderPath: String): Unit = {
    val path = Paths.get(folderPath)
    if (!Files.exists(path)) {
      Files.createDirectories(path)
      logger.info(s"Folder created at: $folderPath")
    } else {
      logger.info(s"Folder already exists at: $folderPath")
    }
  }
}
