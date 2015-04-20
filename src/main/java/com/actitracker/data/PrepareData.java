package com.actitracker.data;


import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PrepareData {

  /**
   * identify Jump
   */
  public static JavaPairRDD<Long[], Long> boudariesDiff(JavaRDD<Long> timestamps, Long firstElement, Long lastElement) {

    JavaRDD<Long> firstRDD = timestamps.filter(record -> record > firstElement);
    JavaRDD<Long> secondRDD = timestamps.filter(record -> record < lastElement);

    // define periods of recording
    return firstRDD.zip(secondRDD)
                   .mapToPair(pair -> new Tuple2<>(new Long[]{pair._1, pair._2}, pair._1 - pair._2));
  }

  public static JavaPairRDD<Long, Long> defineJump(JavaPairRDD<Long[], Long> tsBoundaries) {

    return tsBoundaries.filter(pair -> pair._2 > 100000000)
                       .mapToPair(pair -> new Tuple2<>(pair._1[1], pair._1[0]));
  }

  // (min, max)
  public static List<Long[]> defineInterval(JavaPairRDD<Long, Long> tsJump, Long firstElement, Long lastElement, long windows) {

    List<Long> flatten = tsJump.flatMap(pair -> Arrays.asList(pair._1, pair._2))
                               .sortBy(t -> t, true, 1)
                               .collect();

    int size = flatten.size(); // always even

    List<Long[]> results = new ArrayList<>();
    // init condition
    results.add(new Long[]{firstElement, flatten.get(0), (long) Math.round((flatten.get(0) - firstElement) / windows)});

    for (int i = 1; i < size - 1; i+=2) {
      results.add(new Long[]{flatten.get(i), flatten.get(i + 1), (long) Math.round((flatten.get(i + 1) - flatten.get(i)) / windows)});
    }

    // end condition
    results.add(new Long[]{flatten.get(size - 1), lastElement, (long) Math.round((lastElement - flatten.get(size - 1)) / windows)});

    return results;
  }

}
