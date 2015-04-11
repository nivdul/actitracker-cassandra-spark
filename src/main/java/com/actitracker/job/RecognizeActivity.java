package com.actitracker.job;


import com.actitracker.data.DataManager;
import com.actitracker.data.ExtractFeature;
import com.actitracker.model.DecisionTrees;
import com.actitracker.model.RandomForests;
import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.util.*;

import static com.actitracker.data.ExtractFeature.*;
import static com.datastax.spark.connector.japi.CassandraJavaUtil.*;
import static java.util.Arrays.*;

public class RecognizeActivity {

  private static List<String> ACTIVITIES = asList("Standing", "Jogging", "Walking", "Sitting");

  public static void main(String[] args) {

    // define Spark context
    SparkConf sparkConf = new SparkConf()
                                  .setAppName("Actitracker")
                                  .set("spark.cassandra.connection.host", "127.0.0.1")
                                  .setMaster("local[*]");

    JavaSparkContext sc = new JavaSparkContext(sparkConf);

    // retrieve data from Cassandra and create an CassandraRDD
    CassandraJavaRDD<CassandraRow> cassandraRowsRDD = javaFunctions(sc).cassandraTable("actitracker", "users");

    // TODO issue @link:https://github.com/nivdul/actitracker-cassandra-spark/issues/1

    List<LabeledPoint> labeledPoints = new ArrayList<>();

    for (int i = 1; i < 38; i++) {

      for (String activity: ACTIVITIES) {

        // create bucket of sorted data
        CassandraJavaRDD<CassandraRow> user = cassandraRowsRDD.select("timestamp", "acc_x", "acc_y", "acc_z")
                                                      .where("user_id=? AND activity=?", i, activity)
                                                      .withAscOrder()
                                                      .limit(150L); // define the right number. 100 seems to be fine

        if (user.count() > 0) {
          // transform into vector without timestamp
          JavaRDD<Vector> vectors = DataManager.toVector(user);
          // transform into array
          JavaRDD<double[]> doubles = DataManager.toDouble(user);
          // data with only timestamp and acc
          JavaRDD<long[]> timestamp = DataManager.withTimestamp(user);

          // extract features from this bucket

          // the average acceleration
          ExtractFeature extractFeature = new ExtractFeature(vectors);

          double[] mean = extractFeature.computeAvgAcc();
          System.out.println("Average (mean_x, mean_y, mean_z): " + mean[0] + "," + mean[1] + "," + mean[2]);

          // the variance
          double[] variance = extractFeature.computeVariance();
          System.out.println("Variance (var_x, var_y, var_z): " + variance[0] + "," + variance[1] + "," + variance[2]);

          // the average absolute difference
          double[] avgAbsDiff = computeAvgAbsDifference(doubles, mean);
          System.out.println("Average absolute difference (avg_abs_diff_x, avg_abs_diff_y, avg_abs_diff_z): " + avgAbsDiff[0] + "," + avgAbsDiff[1] + "," + avgAbsDiff[2]);

          // the average resultant acceleration
          double resultant = computeResultantAcc(doubles);
          System.out.println("Average resultant acceleration (res): " + resultant);

          // the average time between peaks
          double avgTimePeak = extractFeature.computeAvgTimeBetweenPeak(timestamp);
          System.out.println("Average time between peaks (peak_y): " + avgTimePeak);


          // build the data set with label & features (11)
          // activity, mean_x, mean_y, mean_z, var_x, var_y, var_z, avg_abs_diff_x, avg_abs_diff_y, avg_abs_diff_z, res, peak_y

          // Let's build LabeledPoint, the structure used in MLlib to create and a predictive model

          // First the features
          double[] features = new double[]{
              mean[0],
              mean[1],
              mean[2],
              variance[0],
              variance[1],
              variance[2],
              avgAbsDiff[0],
              avgAbsDiff[1],
              avgAbsDiff[2],
              resultant,
              avgTimePeak
          };

          // Now the label
          double label = 0;

          if ("Jogging".equals(activity)) {
            label = 1;
          } else if ("Standing".equals(activity)) {
            label = 2;
          } else if ("Sitting".equals(activity)) {
            label = 3;
          }

          LabeledPoint labeledPoint = new LabeledPoint(label, Vectors.dense(features));
          labeledPoints.add(labeledPoint);

        }
      }
    }

    if (labeledPoints.size() > 0) {
      // data ready to be used to build the model
      JavaRDD<LabeledPoint> data = sc.parallelize(labeledPoints);

      // create model prediction and train data on it
      Double meanRF = 0.0;
      Double meanDT = 0.0;

      for (int i = 0; i < 1; i++) {
        // Split data into 2 sets : training (60%) and test (40%).
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.6, 0.4});
        JavaRDD<LabeledPoint> trainingData = splits[0].cache();
        JavaRDD<LabeledPoint> testData = splits[1];

        // With Random Forest
        meanRF += new RandomForests(trainingData, testData).createModel();

        // With DecisionTree
        meanDT += new DecisionTrees(trainingData, testData).createModel();

      }

      System.out.println("sample size " + labeledPoints.size());
      System.out.println("Test Error Random Forest: " + meanRF);
      System.out.println("Test Error Decision Tree: " + meanDT);

    }
  }
}
