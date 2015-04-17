package com.actitracker.data;


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

public class PrepareDataTest {

  private PrepareData prepareData;
  private JavaRDD<Long> ts;
  private JavaSparkContext sc;

  @Before
  public void init() {
    prepareData = new PrepareData();

    SparkConf conf = new SparkConf().setAppName("test extract feature")
        .setMaster("local[*]");

    sc = new JavaSparkContext(conf);

    ts = sc.textFile("data/test2_ts.csv", 1).map(line -> Long.valueOf(line));
  }

  @Test
  public void compute_boudaries_and_diff() {
    // Run
    JavaPairRDD<Long[], Long> result = prepareData.boudariesDiff(ts);
    // Assert
    assertEquals(18, ts.count());
    assertEquals(17, result.count());
    assertEquals(20000000, (long) result.first()._1[0]);
    assertEquals(10000000, (long) result.first()._1[1]);
    assertEquals(10000000, (long) result.first()._2);

    assertEquals(40000000, (long) result.take(3).get(1)._1[0]);
    assertEquals(20000000, (long) result.take(3).get(1)._1[1]);
    assertEquals(20000000, (long) result.take(3).get(1)._2);

    assertEquals(50000000, (long) result.take(3).get(2)._1[0]);
    assertEquals(40000000, (long) result.take(3).get(2)._1[1]);
    assertEquals(10000000, (long) result.take(3).get(2)._2);

  }

  @Test public void define_jump() {
    // Init
    JavaPairRDD<Long[], Long> boundariesDiff = prepareData.boudariesDiff(ts);
    // Run
    JavaPairRDD<Long, Long> result = prepareData.defineJump(boundariesDiff);
    // Assert
    assertEquals(4, result.count());

    assertEquals(50000000, (long) result.take(4).get(0)._1);
    assertEquals(160000000, (long) result.take(4).get(0)._2);

    assertEquals(190000000, (long) result.take(4).get(1)._1);
    assertEquals(300000000, (long) result.take(4).get(1)._2);

    assertEquals(360000000, (long) result.take(4).get(2)._1);
    assertEquals(480000000, (long) result.take(4).get(2)._2);

    assertEquals(490000000, (long) result.take(4).get(3)._1);
    assertEquals(600000000, (long) result.take(4).get(3)._2);

  }

  @Test public void define_interval() {
    // Init
    Long firstElement = ts.first();
    Long lastElement = ts.sortBy(t -> t, false, 1).first();

    JavaPairRDD<Long[], Long> boundariesDiff = prepareData.boudariesDiff(ts);

    JavaPairRDD<Long, Long> jump = prepareData.defineJump(boundariesDiff);
    // Run
    List<Long[]> result = prepareData.defineInterval(jump, firstElement, lastElement);
    // Assert
    assertEquals(5, result.size());

    assertEquals(10000000, (long) result.get(0)[0]);
    assertEquals(50000000, (long) result.get(0)[1]);

    assertEquals(160000000, (long) result.get(1)[0]);
    assertEquals(190000000, (long) result.get(1)[1]);

    assertEquals(300000000, (long) result.get(2)[0]);
    assertEquals(360000000, (long) result.get(2)[1]);

    assertEquals(480000000, (long) result.get(3)[0]);
    assertEquals(490000000, (long) result.get(3)[1]);

    assertEquals(600000000, (long) result.get(4)[0]);
    assertEquals(640000000, (long) result.get(4)[1]);
  }

  @Test public void compute_nb_windows_by_interval_recording() {
    // Init
    Long firstElement = ts.first();
    Long lastElement = ts.sortBy(t -> t, false, 1).first();

    JavaPairRDD<Long[], Long> boundariesDiff = prepareData.boudariesDiff(ts);

    JavaPairRDD<Long, Long> jump = prepareData.defineJump(boundariesDiff);

    List<Long[]> intervals = prepareData.defineInterval(jump, firstElement, lastElement);
    // Run
    JavaRDD<Long[]> result = prepareData.computeNbWindowsByInterval(intervals, sc, 30000000);
    // Assert
    assertEquals(4, result.count());

    assertEquals(10000000, (long) result.take(4).get(0)[0]);
    assertEquals(50000000, (long) result.take(4).get(0)[1]);
    assertEquals(1, (long) result.take(4).get(0)[2]);

    assertEquals(160000000, (long) result.take(4).get(1)[0]);
    assertEquals(190000000, (long) result.take(4).get(1)[1]);
    assertEquals(1, (long) result.take(4).get(1)[2]);

    assertEquals(300000000, (long) result.take(4).get(2)[0]);
    assertEquals(360000000, (long) result.take(4).get(2)[1]);
    assertEquals(2, (long) result.take(4).get(2)[2]);

    assertEquals(600000000, (long) result.take(4).get(3)[0]);
    assertEquals(640000000, (long) result.take(4).get(3)[1]);
    assertEquals(1, (long) result.take(4).get(3)[2]);

  }
}
