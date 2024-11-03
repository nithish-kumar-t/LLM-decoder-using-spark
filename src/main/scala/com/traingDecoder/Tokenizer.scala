package com.traingDecoder

import scala.collection.mutable

/**
 * Tokenizer is a utility class for tokenizing text data in a format suitable for machine learning models.
 * It maps words to unique indices and allows encoding text to sequences of indices and decoding sequences back to text.
 *
 * Usage:
 * - Use `fit` to initialize the vocabulary with a set of training texts.
 * - Use `encode` to convert text into indexed sequences for model training.
 * - Use `decode` to convert generated sequences of indices back into readable text.
 */
class Tokenizer extends Serializable {
  // Using mutable map because this is used to store the vocab, mapping vocab to Indexes

  private val wordToIndex: mutable.Map[String, Int] = mutable.Map[String, Int]()
  private val indexToWord: mutable.Map[Int, String] = mutable.Map[Int, String]()
  private var currentIdx = 0

  /**
   * Processes a sequence of texts to build a vocabulary. It assigns a unique index to each
   * distinct word, creating mappings in `wordToIndex` and `indexToWord` for encoding and decoding.
   *
   * @param texts A sequence of text strings to build the vocabulary from.
   */
  def fit(texts: Seq[String]): Unit = {
    texts.flatMap(_.split("\\s+")).distinct.foreach { word =>
      if (!wordToIndex.contains(word)) {
        wordToIndex(word) = currentIdx
        indexToWord(currentIdx) = word
        currentIdx += 1
      }
    }
  }

  /**
   * Converts a string of text into a sequence of integers representing each word's
   * index. If a word is not in the vocabulary, it assigns a default index of 0.
   *
   * @param text A string of text to encode into word indices.
   * @return A sequence of integers representing the indices of words in the text.
   */
  def encode(text: String): Seq[Int] = {
    text.split("\\s+").map(word => wordToIndex.getOrElse(word, 0))
  }

  /**
   * Converts a sequence of indices back into a readable text string based on the vocabulary.
   * If an index is not in the vocabulary, it maps it to an empty string.
   *
   * @param indices A sequence of word indices to decode.
   * @return A decoded string of text.
   */
  def decode(indices: Seq[Int]): String = {
    indices.map(idx => indexToWord.getOrElse(idx, "")).mkString(" ")
  }
}
