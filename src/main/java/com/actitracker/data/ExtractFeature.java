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
 * More about the a study: http://www.cis.fordham.edu/wisdm/includes/files/sensorKDD-2010.pdf
 */
public class ExtractFeature {

  private MultivariateStatisticalSummary summary;

  public ExtractFeature(JavaRDD<Vector> data) {
    this.summary = Statistics.colStats(data.rdd());
  }

  /**
   * @return Vector (mean_acc_x, mean_acc_y, mean_acc_z)
   */
  public Vector computeAvgAcc() {
    return this.summary.mean();
  }

  /**
   * @return Vector (var_acc_x, var_acc_y, var_acc_z)
   */
  public Vector computeVariance() {
    return this.summary.variance();
  }

  /**
   * @return Vector [ (1 / n ) * ∑ |b - mean_b|, for b in {x,y,z} ]
   */
  public static Vector computeAvgAbsDifference(JavaRDD<Double[]> data, Vector mean) {

    // then for each point x compute x - mean
    // then apply an absolute value: |x - mean|
    JavaRDD<Vector> abs = data.map(record -> new double[]{Math.abs(record[0] - mean.toArray()[0]),
        Math.abs(record[1] - mean.toArray()[1]),
        Math.abs(record[2] - mean.toArray()[2])})
                              .map(line -> Vectors.dense(line));

    // And to finish apply the mean: for each axis (1 / n ) * ∑ |b - mean|
    return Statistics.colStats(abs.rdd()).mean();

  }

  /**
   * @return Double resultant = 1/n * ∑ √(x² + y² + z²)
   */
  public static Double computeResultantAcc(JavaRDD<Double[]> data) {
    // first let's compute the square of each value and the sum
    // compute then the root square: √(x² + y² + z²)
    // to finish apply a mean function: 1/n * sum [√(x² + y² + z²)]
    JavaRDD<Vector> squared = data.map(record -> Math.pow(record[0], 2)
                                               + Math.pow(record[1], 2)
                                               + Math.pow(record[2], 2))
                                  .map(sum -> Math.sqrt(sum))
                                  .map(sum -> Vectors.dense(new double[]{sum}));

    return Statistics.colStats(squared.rdd()).mean().toArray()[0];

  }

  public Vector computeAvgTimeBetweenPeak() {
    // TODO LPR
    return null;
  }


}
