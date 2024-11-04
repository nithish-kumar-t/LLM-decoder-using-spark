# Building an Large-Language-Model (LLM) Decoder using Spark



### Author: Nithish Kumar Thathaiahkalva
<!-- ### UIN :  -->
### Email: nthat@uic.edu

##  Description
This project trains a language model using distributed processing in **Apache Spark** with **DeepLearning4J (DL4J)** for large-scale text generation. The model generates text based on a provided seed input, with all training and inference operations optimized for Spark's parallel processing.
This project aims to implement an LLM decoder using a neural network library and to train the model in the cloud with parallel distributed computations in Apache Spark. The trained model will be utilized for text generation and the application will be deployed on AWS EMR.

[Youtube Video Link](https://youtu.be/iClpUkGiGSQ)


## Project structure

### This Project is continuation of HomeWork-1 where we trained the model and we generated vector embeddings.

### 1. Environment and Configuration Setup
   - Loads environment-specific configurations for training and text generation, including settings such as the number of epochs, batch size, learning rate, input/output paths, etc.
   - Supports configuration for both local (local file system) and cloud (Amazon S3) environments.

### 2. Data Loading and Preparation
   - Loads the input text data into an RDD (Resilient Distributed Dataset) in Spark.
   - Applies preprocessing steps to trim whitespace, filter out empty lines, and cache the data to optimize performance.
   - For cloud environments, reads data directly from S3 buckets; otherwise, reads from the local file system.
   - Dataset used for training the model [Wikipedia Text](https://huggingface.co/datasets/Daniel-Saeedi/wikipedia/blob/main/wikipedia-10.txt).

   **Intermediate Result**: A preprocessed, cached RDD containing lines of text, ready for tokenization and training.

### 3. Tokenizer Setup
   - The `Tokenizer` class tokenizes text into sequences of indices, mapping each word to a unique integer.
   - This encoding prepares the text data for feeding into the neural network.

   **Intermediate Result**: A `Tokenizer` instance that can encode words into integer sequences and decode integer sequences back into words.

### 4. Training Data Splitting
   - Splits the tokenized RDD into training and validation sets.
   - Generates training samples by creating context windows (sequences of words) and corresponding target words using a sliding window approach.

   **Intermediate Result**: Two RDDs (one for training and another for validation) with samples consisting of input sequences (context windows) and their corresponding target words, 80% is used for training and 20% for validating.

### 5. Model Initialization and Serialization
   - Builds and initializes a neural network model (MultiLayerNetwork) with a specific configuration.
   - Serializes the model into a byte array format for broadcasting across Spark workers.

   **Intermediate Result**: A serialized version of the initialized model, ready to be distributed for training.

### 6. Distributed Training Process
   - Conducts training in parallel across Spark partitions, with each partition holding a portion of the training samples.
   - For each epoch:
      - Deserializes the model for use within each partition.
      - Processes batches by feeding sequences and targets into the model, which calculates loss and adjusts weights.
      - Logs metrics such as loss, accuracy, and memory usage for each epoch.
      - At the end of each epoch, averages models from all partitions to update the main model.

   **Intermediate Result**: Accumulated metrics (e.g., average loss, accuracy) and an updated model after each epoch. Metrics are stored in a buffer for eventual logging or saving.

### 7. Text Generation using the Trained Model
   - After training, the model generates new sentences based on an initial seed text.
   - Uses a temperature-based sampling method for controlled randomness in word selection.

   **Intermediate Result**: Generated sentences based on the seed text, stored as a single string.

### 8. Saving Results
   - Saves all intermediate metrics (stored in a buffer) and the generated text to the specified output location.
   - Writes results to S3 in cloud environments or to the local file system otherwise.

   **Final Output**:
   - A CSV file with metrics across epochs (e.g., loss, accuracy, epoch duration).
   - A text file with the generated sentences based on the seed text.

<img width="1286" alt="image" src="https://github.com/user-attachments/assets/e79f3e3c-3ad0-4773-b7fe-c285114fe46e">




## Getting Started

### Prerequisites
- **Apache Spark**
- **DeepLearning4J**
- **Java**
- **Amazon S3** (passkeys are required for doing S3 file IO from local environment)

### Usage
Run this project with Spark-submit, passing the environment and input seed text file as arguments.

```bash
spark-submit --class TextGenerationInLLM \      
  --master "local[*]" \
  --executor-memory 4G \
  --total-executor-cores 4 \
  target/scala-2.12/LLM-hw2-assembly-0.1.0-SNAPSHOT.jar
  env=local src/main/resources/input/seed.txt
```


### Environment
OS: Mac

### IDE: IntelliJ IDEA 2022.2.3 (Ultimate Edition)

### SCALA Version: 2.12.18

[//]: # (SBT Version: 1.10.3)

### Spark Version: 3.5.3

Running the test file
Test Files can be found under the directory src/test

````
sbt clean compile test
````

## Running the project in local.

1) Clone this repository
```
git clone git@github.com:nithish-kumar-t/LLM-decoder-using-spark.git
```


2) cd to the Project
```
cd LLM-decoder-using-spark
```
3) update the jars
```
sbt clean update
```

4) Create fat jat using assembly
```
sbt assembly
# This will create a fat Jar
```

5) we can then run UT's and FT's using below
```
sbt test
```

6) SBT application can contain multiple mains, this project has 2, so to check the correct main
```
➜LLM-decoder-using-spark git:(feature) ✗ sbt
[info] started sbt server
sbt:LLM-hw2-jar>
sbt:LLM-hw2> show discoveredMainClasses
[info] * com.traingDecoder.TextGenerationInLLM
[success] Total time: 2 s, completed Nov 3, 2024, 6:12:14 PM
```

7) Create fat jat using assembly
```
eg:
spark-submit --class TextGenerationInLLM \      
  --master "local[*]" \
  --executor-memory 4G \
  --total-executor-cores 4 \
  target/scala-2.12/LLM-hw2-assembly-0.1.0-SNAPSHOT.jar
  env=local src/main/resources/input/seed.txt
```

## Running the Project in AWS

1) Create a Library structure like below in S3

 <img width="244" alt="image" src="https://github.com/user-attachments/assets/187d514d-0309-47f1-91ca-e031e87b1936">


2) Start a new cluster in AWS EMR, use default configuration,


3) After a cluster is created open the cluster and we can add our Spark-Job job as steps. It will show like below select Spark-Application and select the Jar from your s3 and give env values.

<img width="810" alt="image" src="https://github.com/user-attachments/assets/1b9117b8-b8c5-4374-8677-9f85e885be63">


4) We can also able to chain the steps, once you add step it will start running, based on the order


5) Once the Job completes, output will be available under s3, as per pre-configuration in step 1

<img width="868" alt="image" src="https://github.com/user-attachments/assets/110d0de4-67ce-4618-8d83-e19610e13c6b">
<img width="1216" alt="image" src="https://github.com/user-attachments/assets/9115dd8c-d0d0-4b0a-9afc-cf69ad6ec0ba">





## Prerequisites

1. **SPARK**: Set up Spark on your local machine or cluster.

2. **AWS Account**: Create an AWS account and familiarize yourself with AWS EMR.

3. **Deeplearning4j Library**: Ensure that you have the Java Deeplearning4j Library integrated into your project for getting the model.

4. **Scala, Java and Spark**: Make sure Scala, Java and Hadoop (Scala 2.12.18, Java 11.0.25, spatk: 3.5.3)  are installed and configured correctly.

5. **Git and GitHub**: Use Git for version control and host your project repository on GitHub.

6. **IDE**: Use an Integrated Development Environment (IDE) for coding and development.



## Usage

Follow these steps to execute the project:

1. **Data Gathering**: Ensure you have selected a dataset and it will be ready to get processed by you Spark jobs.

2. **Configuration**: Set up the necessary configuration parameters and input file paths.

3. **Spark Execution**:

   a. Run the TextGenerationInLLM job to train the model with data that takes care of the contect and position, It will 

   b. Run the LLMEncoder job to create vector embeddings, it will create vector embedings and write that in a file.

4. **Results**: Examine the results obtained from the MapReduce jobs.

   a. Training info will be stored in a csv file src/main/resources/output/OUT-{TIME}/training_metrics.csv

   b. Generated text will be stored heresrc/main/resources/output/OUT-{TIME}/generated-data.txt

5. **Deployment on AWS EMR**: If required, deploy the project on AWS EMR to train more data.



## Unit / Regression Testing

**Code coverage report**


<img width="520" alt="image" src="https://github.com/user-attachments/assets/130d452b-ab66-4bbf-be84-ee08581e3b2b">



