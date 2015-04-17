package com.actitracker.job;


import com.actitracker.data.DataManager;
import com.actitracker.data.ExtractFeature;
import com.actitracker.data.PrepareData;
import com.actitracker.model.DecisionTrees;
import com.datastax.spark.connector.japi.CassandraRow;
import com.datastax.spark.connector.japi.rdd.CassandraJavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
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
                                  .setAppName("User's physical activity recognition")
                                  .set("spark.cassandra.connection.host", "127.0.0.1")
                                  .setMaster("local[*]");

    JavaSparkContext sc = new JavaSparkContext(sparkConf);

    // retrieve data from Cassandra and create an CassandraRDD
    CassandraJavaRDD<CassandraRow> cassandraRowsRDD = javaFunctions(sc).cassandraTable("actitracker", "users");

    List<LabeledPoint> labeledPoints = new ArrayList<>();

    for (int i = 1; i < 2; i++) {

      for (String activity: ACTIVITIES) {

        // create bucket of sorted data by ascending timestamp
        CassandraJavaRDD<CassandraRow> times = cassandraRowsRDD.select("timestamp")
                                                                  .where("user_id=? AND activity=?", i, activity)
                                                                  .withAscOrder();

        // if data
        if (times.count() > 0) {

          ////////////////////
          // DEFINE THE INTERVALS ON WHICH EXTRACT WINDOWS
          // The data sets provide data from 37 different users. And each user perform different activities several times.
          // So I have defined several windows for each user and each activity to retrieve more samples.
          ////////////////////

          // first find jumps to define the continuous periods of data

          //retrieve all timestamp
          JavaRDD<Long> ts = times.map(CassandraRow::toMap)
                                  .map(entry -> (long) entry.get("timestamp"));

          Long firstElement = ts.first();
          Long lastElement = ts.sortBy(time -> time, false, 1).first();

          // compute the difference between each timestamp
          JavaPairRDD<Long[], Long> tsBoundariesDiff = PrepareData.boudariesDiff(ts);

          // define periods of recording
          // if the difference is greater than 100 000 000, it must be different periods of recording
          // ({min_boundary, max_boundary}, max_boundary - min_boundary > 100 000 000)
          JavaPairRDD<Long, Long> jumps = PrepareData.defineJump(tsBoundariesDiff);

          // Now define interval
          List<Long[]> intervals = PrepareData.defineInterval(jumps, firstElement, lastElement, 5000000000L);

          for (Long[] interval: intervals) {
            for (int j = 0; j < interval[2]; j++) {

              CassandraJavaRDD<CassandraRow> user = cassandraRowsRDD.select("timestamp", "acc_x", "acc_y", "acc_z")
                  .where("user_id=? AND activity=? AND timestamp < ? AND timestamp > ?", i, activity, interval[1] + j * 5000000000L, interval[1] + (j - 1) * 5000000000L)
                  .withAscOrder();

              if (user.count() > 0) {
                // transform into vector without timestamp
                JavaRDD<Vector> vectors = DataManager.toVector(user);
                // transform into array
                JavaRDD<double[]> doubles = DataManager.toDouble(user);
                // data with only timestamp and acc
                JavaRDD<long[]> timestamp = DataManager.withTimestamp(user);

                ///////////////////////////////////////
                // extract features from this bucket //
                ///////////////////////////////////////

                // the average acceleration
                ExtractFeature extractFeature = new ExtractFeature(vectors);

                double[] mean = extractFeature.computeAvgAcc();
                //System.out.println("Average (mean_x, mean_y, mean_z): " + mean[0] + "," + mean[1] + "," + mean[2]);

                // the variance
                double[] variance = extractFeature.computeVariance();
                //System.out.println("Variance (var_x, var_y, var_z): " + variance[0] + "," + variance[1] + "," + variance[2]);

                // the average absolute difference
                double[] avgAbsDiff = computeAvgAbsDifference(doubles, mean);
                //System.out.println("Average absolute difference (avg_abs_diff_x, avg_abs_diff_y, avg_abs_diff_z): " + avgAbsDiff[0] + "," + avgAbsDiff[1] + "," + avgAbsDiff[2]);

                // the average resultant acceleration
                double resultant = computeResultantAcc(doubles);
                //System.out.println("Average resultant acceleration (res): " + resultant);

                // the average time between peaks
                double avgTimePeak = extractFeature.computeAvgTimeBetweenPeak(timestamp);
                //System.out.println("Average time between peaks (peak_y): " + avgTimePeak);


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
        }
      }
    }

    System.out.println("labeledPoints " + labeledPoints.size());

    if (labeledPoints.size() > 0) {
      // data ready to be used to build the model
      JavaRDD<LabeledPoint> data = sc.parallelize(labeledPoints);

      // create model prediction and train data on it

      // Split data into 2 sets : training (60%) and test (40%).
      JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.6, 0.4});
      JavaRDD<LabeledPoint> trainingData = splits[0].cache();
      JavaRDD<LabeledPoint> testData = splits[1];

      // With Random Forest
      //double errRF = new RandomForests(trainingData, testData).createModel();

      // With DecisionTree
      double errDT = new DecisionTrees(trainingData, testData).createModel();

      System.out.println("sample size " + labeledPoints.size());
      //System.out.println("Test Error Random Forest: " + errRF);
      System.out.println("Test Error Decision Tree: " + errDT);

    }
  }
}
