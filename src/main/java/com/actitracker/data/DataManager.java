package com.actitracker.data;


import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class DataManager {

  public static JavaRDD<Vector> toVector(CassandraJavaRDD<CassandraRow> data) {

    // first transform CassandraRDD into a RDD<Map>
    return data.map(cassandraRow -> cassandraRow.toMap())
                // then build  a double array from the RDD<Map>
                .map(entry -> new double[]{(double) entry.get("acc_x"), (double) entry.get("acc_y"), (double) entry.get("acc_z")})
               // to finish apply the dense method
               .map(line -> Vectors.dense(line));

  }

  public static JavaRDD<Double[]> toDouble(CassandraJavaRDD<CassandraRow> data) {

    // first transform CassandraRDD into a RDD<Map>
    return data.map(cassandraRow -> cassandraRow.toMap())
        // then build  a double array from the RDD<Map>
        .map(entry -> new Double[]{(Double) entry.get("acc_x"), (Double) entry.get("acc_y"), (Double) entry.get("acc_z")});

  }

}
