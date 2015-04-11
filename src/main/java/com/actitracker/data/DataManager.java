package com.actitracker.data;


import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class DataManager {

  public static JavaRDD<Vector> toVector(CassandraJavaRDD<CassandraRow> data) {

    // first transform CassandraRDD into a RDD<Map>
    return data.map(CassandraRow::toMap)
                // then build  a double array from the RDD<Map>
                .map(entry -> new double[]{(double) entry.get("acc_x"), (double) entry.get("acc_y"), (double) entry.get("acc_z")})
               // to finish apply the dense method
               .map(Vectors::dense);
  }

  public static JavaRDD<double[]> toDouble(CassandraJavaRDD<CassandraRow> data) {

    // first transform CassandraRDD into a RDD<Map>
    return data.map(CassandraRow::toMap)
        // then build  a double array from the RDD<Map>
        .map(entry -> new double[]{(Double) entry.get("acc_x"), (Double) entry.get("acc_y"), (Double) entry.get("acc_z")});
  }

  public static JavaRDD<long[]> withTimestamp(CassandraJavaRDD<CassandraRow> data) {

    // first transform CassandraRDD into a RDD<Map>
    return data.map(CassandraRow::toMap)
        // then build  a double array from the RDD<Map>
        .map(entry -> new long[]{(Long) entry.get("timestamp"), ((Double) entry.get("acc_y")).longValue()});
  }

}
