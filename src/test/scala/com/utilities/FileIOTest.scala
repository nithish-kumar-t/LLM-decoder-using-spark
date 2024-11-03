package com.utilities

import org.scalatest.BeforeAndAfter
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.must.Matchers
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

import java.nio.file.{Files, Path, Paths}

class FileIOTest extends AnyFlatSpec with Matchers with BeforeAndAfter {
  val directoryPath = "/Users/tnithish/Learning/CS-441/LLM-decoder-using-spark/src/main/resources/test"
  val directory: Path = Paths.get(directoryPath)

  val testFilePath = s"$directoryPath/input-text.txt"
  val testFolderPath = s"$directoryPath/testFolder"
  val testContent = "This is a test content."


  "FileIO" should "correctly write and read a file" in {
    // Write content to file
    FileIO.writeStringToFileNIO(testFilePath, testContent)

    // Verify the file exists and content is correct
    Files.exists(Paths.get(testFilePath)) shouldBe true
    val content = FileIO.readFileToStringNIO(testFilePath)
    content shouldBe testContent
  }

  it should "not create the folder if it already exists" in {
    // Create the folder
    FileIO.createFolder(testFolderPath)
    Files.exists(Paths.get(testFolderPath)) shouldBe true

    // Attempt to create the folder again and check no error occurs
    noException should be thrownBy FileIO.createFolder(testFolderPath)
    Files.isDirectory(Paths.get(testFolderPath)) shouldBe true

  }

  it should "overwrite existing file content on write" in {
    // Write initial content
    FileIO.writeStringToFileNIO(testFilePath, "Initial content")

    // Write new content
    FileIO.writeStringToFileNIO(testFilePath, testContent)

    // Verify content has been overwritten
    val content = FileIO.readFileToStringNIO(testFilePath)
    content shouldBe testContent
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

