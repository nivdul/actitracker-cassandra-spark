package com.actitracker.data;


import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class DataManager {

  public static JavaRDD<Vector> toVector(CassandraJavaRDD<CassandraRow> data) {

    // first transform CassandraRDD into a RDD<Map>
    // then build  a double array from the RDD<Map>
    // to finish apply the dense method
    return data.map(cassandraRow -> cassandraRow.toMap())
               .map(entry -> new double[]{(double) entry.get("acc_x"), (double) entry.get("acc_y"), (double) entry.get("acc_z")})
               .map(line -> Vectors.dense(line));

  }
}
