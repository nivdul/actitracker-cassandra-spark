package com.actitracker.data;


import com.datastax.spark.connector.japi.CassandraRow;
import org.apache.spark.api.java.JavaRDD;

public class DataManager {

  public static JavaRDD<double[]> toDouble(JavaRDD<CassandraRow> data) {

    // first transform CassandraRDD into a RDD<Map>
    return data.map(CassandraRow::toMap)
        // then build  a double array from the RDD<Map>
        .map(entry -> new double[]{(double) entry.get("acc_x"), (double) entry.get("acc_y"), (double) entry.get("acc_z")});
  }

  public static JavaRDD<long[]> withTimestamp(JavaRDD<CassandraRow> data) {

    // first transform CassandraRDD into a RDD<Map>
    return data.map(CassandraRow::toMap)
        // then build  a double array from the RDD<Map>
        .map(entry -> new long[]{(long) entry.get("timestamp"), ((Double) entry.get("acc_y")).longValue()});
  }

}
