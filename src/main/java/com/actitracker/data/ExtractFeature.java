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
 * - Average resultant acceleration: 1/n * ∑ √(x² + y² + z²)
 * - Average time between peaks (max) (for each axis)
 *
 */
public class ExtractFeature {

  private MultivariateStatisticalSummary summary;

  public ExtractFeature(JavaRDD<Vector> data) {
    this.summary = Statistics.colStats(data.rdd());
  }

  /**
   * @return array (mean_acc_x, mean_acc_y, mean_acc_z)
   */
  public double[] computeAvgAcc() {
    return this.summary.mean().toArray();
  }

  /**
   * @return array (var_acc_x, var_acc_y, var_acc_z)
   */
  public double[] computeVariance() {
    return this.summary.variance().toArray();
  }

  /**
   * @return array [ (1 / n ) * ∑ |b - mean_b|, for b in {x,y,z} ]
   */
  public static double[] computeAvgAbsDifference(JavaRDD<double[]> data, double[] mean) {

    // for each point x compute x - mean
    // then apply an absolute value: |x - mean|
    JavaRDD<Vector> abs = data.map(record -> new double[]{Math.abs(record[0] - mean[0]),
                                                          Math.abs(record[1] - mean[1]),
                                                          Math.abs(record[2] - mean[2])})
                              .map(Vectors::dense);

    // And to finish apply the mean: for each axis (1 / n ) * ∑ |b - mean|
    return Statistics.colStats(abs.rdd()).mean().toArray();

  }

  /**
   * @return Double resultant = 1/n * ∑ √(x² + y² + z²)
   */
  public static double computeResultantAcc(JavaRDD<double[]> data) {
    // first let's compute the square of each value and the sum
    // compute then the root square: √(x² + y² + z²)
    // to finish apply a mean function: 1/n * sum [√(x² + y² + z²)]
    JavaRDD<Vector> squared = data.map(record -> Math.pow(record[0], 2)
                                               + Math.pow(record[1], 2)
                                               + Math.pow(record[2], 2))
                                  .map(Math::sqrt)
                                  .map(sum -> Vectors.dense(new double[]{sum}));

    return Statistics.colStats(squared.rdd()).mean().toArray()[0];

  }

  /**
   * compute average time between peaks.
   */
  public Double computeAvgTimeBetweenPeak(JavaRDD<long[]> data) {
    // define the maximum
    double[] max = this.summary.max().toArray();

    // keep the timestamp of data point for which the value is greater than 0.9 * max
    // and sort it !
    JavaRDD<Long> filtered_y = data.filter(record -> record[1] > 0.9 * max[1])
                                   .map(record -> record[0])
                                   .sortBy(time -> time, true, 1);

    if (filtered_y.count() > 1) {
      Long firstElement = filtered_y.first();
      Long lastElement = filtered_y.sortBy(time -> time, false, 1).first();

      // compute the delta between each tick
      JavaRDD<Long> firstRDD = filtered_y.filter(record -> record > firstElement);
      JavaRDD<Long> secondRDD = filtered_y.filter(record -> record < lastElement);

      JavaRDD<Vector> product = firstRDD.zip(secondRDD)
                                        .map(pair -> pair._1() - pair._2())
                                         // and keep it if the delta is != 0
                                        .filter(value -> value > 0)
                                        .map(line -> Vectors.dense(line));

      // compute the mean of the delta
      return Statistics.colStats(product.rdd()).mean().toArray()[0];
    }

    return 0.0;
  }

}