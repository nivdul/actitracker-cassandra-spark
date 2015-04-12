package com.actitracker.data;


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class ExtractFeatureTest {

  private ExtractFeature extractFeature;
  private JavaRDD<double[]> data;
  private JavaRDD<long[]> data_with_ts;

  @Before
  public void init() {
    SparkConf conf = new SparkConf().setAppName("test extract feature")
        .setMaster("local[*]");

    JavaSparkContext sc = new JavaSparkContext(conf);

    data = sc.textFile("data/test.csv")
                             .map(line -> line.split(","))
                             .map(line -> new double[]{Double.valueOf(line[0]), Double.valueOf(line[1]), Double.valueOf(line[2])});

    data_with_ts = sc.textFile("data/test_with_ts.csv")
        .map(line -> line.split(","))
        .map(line -> new long[]{Long.valueOf(line[0]), Double.valueOf(line[1]).longValue()});

    extractFeature = new ExtractFeature(data.map(Vectors::dense));

  }

  @Test
  public void mean() {
    // run
    double[] mean = extractFeature.computeAvgAcc();
    // assert
    Assert.assertTrue(-5.23 == Math.round(mean[0]*100)/100.0);
    Assert.assertTrue(8.11 == Math.round(mean[1]*100)/100.0);
    Assert.assertTrue(1.22 == Math.round(mean[2]*100)/100.0);
  }

  @Test
  public void variance() {
    // run
    double[] var = extractFeature.computeVariance();
    // assert
    Assert.assertTrue(0.02 == Math.round(var[0]*100)/100.0);
    Assert.assertTrue(0.008 == Math.round(var[1]*1000)/1000.0);
    Assert.assertTrue(0.01 == Math.round(var[2]*100)/100.0);
  }

  @Test
  public void average_absolute_difference() {
    // init
    double[] mean = extractFeature.computeAvgAcc();
    // run
    double[] avgAbsDiff = ExtractFeature.computeAvgAbsDifference(data, mean);
    // assert
    Assert.assertTrue(0.12 == Math.round(avgAbsDiff[0]*100)/100.0);
    Assert.assertTrue(0.06 == Math.round(avgAbsDiff[1]*100)/100.0);
    Assert.assertTrue(0.10 == Math.round(avgAbsDiff[2]*100)/100.0);
  }

  @Test
  public void resultant_acceleration() {
    // run
    double resultant = ExtractFeature.computeResultantAcc(data);
    // assert
    Assert.assertTrue(9.73 == Math.round(resultant*100)/100.0);
  }

  // TODO fix it !
  @Test
  public void average_time_between_peaks_y() {
    // run
    double avgTimePeak = extractFeature.computeAvgTimeBetweenPeak(data_with_ts);
    // assert
    Assert.assertTrue(1.0 == Math.round(avgTimePeak*100)/100.0);
  }
}
