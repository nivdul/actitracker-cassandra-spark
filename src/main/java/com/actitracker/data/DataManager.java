package com.actitracker.data;


import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

public class DataManager {

  public static JavaRDD<Vector> toVector(CassandraJavaRDD<CassandraRow> data) {

    // TODO LPR
    return null;
  }
}
