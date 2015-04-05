package com.actitracker.job;


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

public class RecognizeActivity {

  public static void main(String[] args) {
    // define Spark context
    SparkConf sparkConf = new SparkConf()
        .setAppName("Actitracker")
        .setMaster("local[*]");

    JavaSparkContext sc = new JavaSparkContext(sparkConf);

    // load the data
    JavaRDD<String> data = sc.textFile("data.csv");

    System.out.println(data.first());
    System.out.println(data.count());

  }
}
