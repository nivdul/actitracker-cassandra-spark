package com.actitracker.data;


import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;

/**
 * We use labeled accelerometer data from users thanks to a device in their pocket during different activities (walking, sitting, jogging, ascending stairs, descending stairs, and standing).
 *
 * The accelerometer measures acceleration in all three spatial dimensions as following:
 *
 * - Z-axis captures the forward movement of the leg
 * - Y-axis captures the upward and downward movement of the leg
 * - X-axis captures the horizontal movement of the leg
 *
 * After several tests with different features combination, the ones that I have chosen are described below:
 *
 * - Average acceleration (for each axis)
 * - Variance (for each axis)
 * - Average absolute difference (for each axis)
 * - Average resultant acceleration: 1/n * ∑√(x² + y² + z²)
 * - Average time between peaks (max) (for each axis)
 */
public class ExtractFeature {

  private MultivariateStatisticalSummary summary;

  public ExtractFeature(JavaRDD<Vector> data) {
    this.summary = Statistics.colStats(data.rdd());
  }

  public Vector computeAvgAcc() {
    return this.summary.mean();
  }

  public Vector computeVariance() {
    return this.summary.variance();
  }

  public static Vector computeAvgAbsDifference(JavaRDD<Vector> data, Vector mean) {

    // then for each point x compute x - mean
    // then apply an absolute value: |x - mean|
    JavaRDD<Vector> abs = data.map(vector -> new double[]{Math.abs(vector.toArray()[0] - mean.toArray()[0]),
                                                          Math.abs(vector.toArray()[1] - mean.toArray()[1]),
                                                          Math.abs(vector.toArray()[2] - mean.toArray()[2])})
                              .map(line -> Vectors.dense(line));

    // And to finish compute the mean: (1 / n ) * ∑ |b - mean|, for each axis
    return Statistics.colStats(abs.rdd()).mean();

  }

  public Double computeResultantAcc() {
    // TODO LPR
    return 0.0;
  }

  public Vector computeAvgTimeBetweenPeak() {
    // TODO LPR
    return null;
  }


}
