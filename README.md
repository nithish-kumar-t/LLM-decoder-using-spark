# Building an Large-Language-Model (LLM) Decoder using Spark



### Author: Nithish Kumar Thathaiahkalva
<!-- ### UIN :  -->
### Email: nthat@uic.edu

##  Description

This project aims to implement an LLM decoder using a neural network library and to train the model in the cloud with parallel distributed computations in Apache Spark. The trained model will be utilized for text generation and the application will be deployed on AWS EMR.


```bash
spark-submit --class WordCountSpark \      
  --master "local[*]" \
  --executor-memory 4G \
  --total-executor-cores 4 \
  target/scala-2.12/LLM-hw1-assembly-0.1.0-SNAPSHOT.jar
```

##  Project structure
