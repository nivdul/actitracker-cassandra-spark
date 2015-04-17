package com.actitracker.data;


import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PrepareData {

  public JavaPairRDD<Long[], Long> boudariesDiff(JavaRDD<Long> timestamps) {

    Long firstElement = timestamps.first();
    Long lastElement = timestamps.sortBy(time -> time, false, 1).first();

    JavaRDD<Long> firstRDD = timestamps.filter(record -> record > firstElement);
    JavaRDD<Long> secondRDD = timestamps.filter(record -> record < lastElement);

    // define periods of recording
    JavaPairRDD<Long[], Long> tsBoundaries = firstRDD.zip(secondRDD)
        .mapToPair(pair -> new Tuple2<>(new Long[]{pair._1, pair._2}, pair._1 - pair._2));

    return tsBoundaries;
  }

  public JavaPairRDD<Long, Long> defineJump(JavaPairRDD<Long[], Long> tsBoundaries) {
    return tsBoundaries.filter(pair -> pair._2 > 100000000)
                       .mapToPair(pair -> new Tuple2<>(pair._1[1], pair._1[0]));
  }

  // (min, max)
  public List<Long[]> defineInterval(JavaPairRDD<Long, Long> tsJump, Long firstElement, Long lastElement) {

    List<Long> flatten = tsJump.flatMap(pair -> Arrays.asList(pair._1, pair._2))
                                  .sortBy(t -> t, true, 1)
                                  .collect();

    int size = flatten.size(); // always even

    List<Long[]> results = new ArrayList<>();
    // init condition
    results.add(new Long[]{firstElement, flatten.get(0)});

    for (int i = 1; i < size - 1; i+=2) {
      results.add(new Long[]{flatten.get(i), flatten.get(i + 1)});
    }

    // end limite
    results.add(new Long[]{flatten.get(size - 1), lastElement});

    return results;
  }

  public JavaRDD<Long[]> computeNbWindowsByInterval(List<Long[]> intervals, JavaSparkContext sc, int windows) {
    return sc.parallelize(intervals)
             .map(pair -> new Long[]{pair[0], pair[1], (long) Math.round((pair[1] - pair[0]) / windows)})
             .filter(line -> 0 != line[2]);
  }
}
