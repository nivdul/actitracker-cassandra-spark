package com.actitracker.job;


import com.actitracker.data.DataManager;
import com.actitracker.data.ExtractFeature;
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
import scala.Tuple2;

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
        CassandraJavaRDD<CassandraRow> times = cassandraRowsRDD.select("timestamp")
                                                                  .where("user_id=? AND activity=?", i, activity)
                                                                  .withAscOrder();

        if (times.count() > 0) {
          ////////////////////
          // define windows //
          ////////////////////

          // first define jump

          //retrieve all timestamp
          JavaRDD<Long> ts = times.map(CassandraRow::toMap)
                                    .map(entry -> (long) entry.get("timestamp"))
                                    .sortBy(time -> time, true, 1);
          // compute the difference between each timestamp
          Long firstElement = ts.first();
          Long lastElement = ts.sortBy(time -> time, false, 1).first();

          JavaRDD<Long> first_ts = ts.filter(record -> record > firstElement);
          JavaRDD<Long> second_ts = ts.filter(record -> record < lastElement);

          JavaPairRDD<Long,Long> ts_jump_70000000 = first_ts.zip(second_ts)
              .mapToPair(pair -> new Tuple2<>(pair._1(), pair._1() - pair._2()))
              .filter(pair -> pair._2() > 70000000);

          // define periods
          Long firstElement_ts_jump = ts_jump_70000000.first()._1();
          Long lastElement_ts_jump = ts_jump_70000000.sortByKey(false).first()._1();

          JavaRDD<Long> first_ts_jump = ts_jump_70000000.filter(record -> record._1() > firstElement_ts_jump).map(pair -> pair._1());
          JavaRDD<Long> second_ts_jump = ts_jump_70000000.filter(record -> record._1() < lastElement_ts_jump).map(pair -> pair._1());

          // start, end, window number
          JavaRDD<Long[]> periodsRDD = first_ts_jump.zip(second_ts_jump)
              .map(tuple -> new Long[]{tuple._1(), tuple._2(), (long) Math.round((tuple._1() - tuple._2()) / 5000000000L)});

          List<Long[]> periods = periodsRDD.collect();

          for (Long[] period: periods) {
            for (int j = 0; j < period[2]; j++) {

              CassandraJavaRDD<CassandraRow> user = cassandraRowsRDD.select("timestamp", "acc_x", "acc_y", "acc_z")
                  .where("user_id=? AND activity=? AND timestamp < ? AND timestamp > ?", i, activity, period[1] + j * 5000000000L, period[1] + (j - 1) * 5000000000L)
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
