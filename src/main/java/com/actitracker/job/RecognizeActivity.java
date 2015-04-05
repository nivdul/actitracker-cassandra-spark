package com.actitracker.job;


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

public class RecognizeActivity {

  public static void main(String[] args) {
    // define Spark context
    SparkConf sparkConf = new SparkConf()
        .setAppName("Actitracker")
        .setMaster("local[*]");

    JavaSparkContext sc = new JavaSparkContext(sparkConf);

    // load the data : user_id, activity, ts, acc_x, acc_y, acc_z
    JavaRDD<String[]> data = sc.textFile("data.csv")
        .filter(line -> line.length() > 1)
        .map(line -> line.substring(0, line.length() - 1))
        .map(line -> line.split(","))
        .filter(line -> line.length == 6)
        .cache();

    System.out.println(data.first()[0]);
    System.out.println(data.first()[1]);
    System.out.println(data.first()[2]);
    System.out.println(data.first()[3]);
    System.out.println(data.first()[4]);
    System.out.println(data.first()[5]);
    System.out.println(data.count());

  }
}
