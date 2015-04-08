package com.actitracker.job;


import com.actitracker.data.DataManager;
import com.actitracker.data.ExtractFeature;
import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;

import static com.datastax.spark.connector.japi.CassandraJavaUtil.*;

public class RecognizeActivity {

  public static void main(String[] args) {
    // define Spark context
    SparkConf sparkConf = new SparkConf()
        .setAppName("Actitracker")
        .set("spark.cassandra.connection.host", "127.0.0.1")
        .setMaster("local[*]");

    JavaSparkContext sc = new JavaSparkContext(sparkConf);

    // retrieve data from Cassandra and create an CassandraRDD
    CassandraJavaRDD<CassandraRow> cassandraRowsRDD = javaFunctions(sc).cassandraTable("actitracker", "users");

    // create bucket of sorted data
    // TODO do it for each user
    int user_id = 8;
    String activity = "Jogging";
    long startTimeStamp = 110821191627000L;

    CassandraJavaRDD<CassandraRow> user = cassandraRowsRDD.select("acc_x", "acc_y", "acc_z")
                                                          .where("user_id=? AND activity=? AND timestamp >=?", user_id, activity, startTimeStamp)
                                                          .withAscOrder()
                                                          .limit(100L);

    // transform array into vector
    JavaRDD<Vector> vectors = DataManager.toVector(user);

    /////////////////////
    // extract features //
    /////////////////////

    // the average acceleration
    ExtractFeature extractFeature = new ExtractFeature(vectors);

    Vector mean = extractFeature.computeAvgAcc();
    System.out.println("Average (acc_x, acc_y, acc_z): " + mean.toArray()[0] + "," + mean.toArray()[1] + "," + mean.toArray()[2]);

    // the variance
    Vector variance = extractFeature.computeVariance();
    System.out.println("Variance (var_x, var_y, var_z): " + variance.toArray()[0] + "," + variance.toArray()[1] + "," + variance.toArray()[2]);

    // the average absolute difference
    Vector avgAbsDiff = ExtractFeature.computeAvgAbsDifference(vectors, mean);
    System.out.println("Average absolute difference (avg_abs_diff_x, avg_abs_diff_y, avg_abs_diff_z): " + avgAbsDiff.toArray()[0] + "," + avgAbsDiff.toArray()[1] + "," + avgAbsDiff.toArray()[2]);

    // the average resultant acceleration: 1/n * ∑√(x² + y² + z²)
    Double resultant = extractFeature.computeResultantAcc();

    // the average time between peaks (max)
    Vector avgTimePeak = extractFeature.computeAvgTimeBetweenPeak();

  }
}
