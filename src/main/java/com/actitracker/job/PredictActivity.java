package com.actitracker.job;


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;

public class PredictActivity {


  public static void main(String[] args) {

    SparkConf sparkConf = new SparkConf()
        .setAppName("User's physical activity recognition")
        .set("spark.cassandra.connection.host", "127.0.0.1")
        .setMaster("local[*]");

    JavaSparkContext sc = new JavaSparkContext(sparkConf);

    System.out.println(predict(sc));

  }

  public static double predict(JavaSparkContext sc) {

    DecisionTreeModel model = DecisionTreeModel.load(sc.sc(), "actitracker");

    double[] feature = {3.3809183673469394,-6.880102040816324,0.8790816326530612,50.08965378708187,84.13105050494424,20.304453787081833,5.930491461890875,7.544194085797583,3.519248229904206,12.968485972481643,7.50031E8};

    Vector sample = Vectors.dense(feature);
    double prediction = model.predict(sample);

    return prediction;

  }
}
