package com.decoder

import org.scalatest.BeforeAndAfter
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.must.Matchers

import java.nio.file.{Files, Paths}

class BaseLLMTest extends AnyFlatSpec with Matchers with BeforeAndAfter{
  // Output directory path
  val directoryPath = "/Users/tnithish/Learning/CS-441/LLM-decoder-using-spark/src/main/resources/output"
  val directory = Paths.get(directoryPath)

  before {
    // Check if the directory exists
    if (Files.exists(directory) && Files.isDirectory(directory)) {
      // List all files in the directory and delete them
      Files.list(directory).forEach { file =>
        if (Files.isRegularFile(file)) { // Check if it is a file
          Files.delete(file) // Delete the file
          println(s"Deleted: ${file.getFileName}")
        } else {
          Files.walk(file).sorted(java.util.Comparator.reverseOrder()).forEach(Files.delete)
        }
      }
    } else {
      println(s"Directory does not exist: $directoryPath")
    }
  }

  after {
    //delete the files in the output directory after the completion of the program
    if (Files.exists(directory) && Files.isDirectory(directory)) {
      Files.list(directory).forEach(file => {
        if (!Files.isRegularFile(file)) {
          Files.walk(file).sorted(java.util.Comparator.reverseOrder()).forEach(Files.delete)
        }
      })
    }
  }

}
