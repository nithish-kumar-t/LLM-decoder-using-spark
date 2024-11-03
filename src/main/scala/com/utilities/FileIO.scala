package com.utilities

import org.slf4j.LoggerFactory

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

object FileIO {
  private val logger = LoggerFactory.getLogger(getClass)

  def readFileToStringNIO(filePath: String): String = {
    Files.readString(Paths.get(filePath)) // Reads the file as a single string
  }

  def writeStringToFileNIO(filePath: String, content: String): Unit = {
    Files.write(Paths.get(filePath), content.getBytes(StandardCharsets.UTF_8))
  }

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
