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

    // retrieve data from Cassandra and creat an CassandraRDD
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
    // extract feature //
    /////////////////////

    // the average acceleration (x,y,z)
    ExtractFeature extractFeature = new ExtractFeature(vectors);

    Vector mean = extractFeature.computeAvgAcc();

    System.out.println("average acc_x: " + mean.toArray()[0]);
    System.out.println("average acc_y: " + mean.toArray()[1]);
    System.out.println("average acc_z: " + mean.toArray()[2]);

    // the variance (x,y,z)
    Vector variance = extractFeature.computeVariance();


  }
}
