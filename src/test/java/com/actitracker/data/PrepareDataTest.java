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

  private JavaRDD<Long> times;
  private JavaSparkContext sc;
  Long firstElement;
  Long lastElement;

  @Before
  public void init() {

    SparkConf conf = new SparkConf().setAppName("test extract feature")
        .setMaster("local[*]");

    sc = new JavaSparkContext(conf);

    times = sc.textFile("data/test2_ts.csv", 1).map(line -> Long.valueOf(line));

    firstElement = times.first();
    lastElement = times.sortBy(time -> time, false, 1).first();

  }

  @Test
  public void compute_boudaries_and_diff() {
    // Run
    JavaPairRDD<Long[], Long> result = PrepareData.boudariesDiff(times, firstElement, lastElement);
    // Assert
    assertEquals(18, times.count());
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
    JavaPairRDD<Long[], Long> boundariesDiff = PrepareData.boudariesDiff(times, firstElement, lastElement);
    // Run
    JavaPairRDD<Long, Long> result = PrepareData.defineJump(boundariesDiff);
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
    JavaPairRDD<Long[], Long> boundariesDiff = PrepareData.boudariesDiff(times, firstElement, lastElement);

    JavaPairRDD<Long, Long> jump = PrepareData.defineJump(boundariesDiff);
    // Run
    List<Long[]> result = PrepareData.defineInterval(jump, firstElement, lastElement, 30000000);
    // Assert
    assertEquals(4, result.size());

    assertEquals(10000000, (long) result.get(0)[0]);
    assertEquals(50000000, (long) result.get(0)[1]);
    assertEquals(1, (long) result.get(0)[2]);

    assertEquals(160000000, (long) result.get(1)[0]);
    assertEquals(190000000, (long) result.get(1)[1]);
    assertEquals(1, (long) result.get(1)[2]);

    assertEquals(300000000, (long) result.get(2)[0]);
    assertEquals(360000000, (long) result.get(2)[1]);
    assertEquals(2, (long) result.get(2)[2]);

    assertEquals(600000000, (long) result.get(3)[0]);
    assertEquals(640000000, (long) result.get(3)[1]);
    assertEquals(1, (long) result.get(3)[2]);
  }

}
