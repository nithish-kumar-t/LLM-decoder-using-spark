package com.decoder

import com.traingDecoder.TextGenerationInLLM
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

import java.nio.file.Files

class TextGenerationLLMTest extends BaseLLMTest {

  "TextGenerationInLLM" should " when incorrect arguments it shouldn't execute further" in {
    TextGenerationInLLM.main(Array("env=test"))
    Files.list(directory).forEach { file =>
      Files.list(file).count() shouldBe (0)
    }
  }

  "TextGenerationInLLM" should " when incorrect env name it shouldn't execute further" in {
    TextGenerationInLLM.main(Array("env=random_env"))
    Files.list(directory).forEach { file =>
      Files.list(file).count() shouldBe (0)
    }
  }

  "TextGenerationInLLM" should " when incorrect seed file path name shouldn't execute further" in {
    TextGenerationInLLM.main(Array("env=random_env", "src/main/resources/input/random-file.txt"))
    Files.list(directory).forEach { file =>
      Files.list(file).count() shouldBe (0)
    }
  }

  "TextGenerationInLLM" should " when correct seed file path and env name should execute without any errors" in {
    TextGenerationInLLM.main(Array("env=test", "src/main/resources/input/seed.txt"))
    Files.list(directory).forEach { file =>
      Files.list(file).count() shouldBe (2)
    }
  }
}

