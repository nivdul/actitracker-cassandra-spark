package com.actitracker.model;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

public class RandomForest {

  JavaRDD<LabeledPoint> trainingData;
  JavaRDD<LabeledPoint> testData;

  public RandomForest(JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData) {
    this.trainingData = trainingData;
    this.testData = testData;
  }

  /**
   * Train a RandomForest model
   */
  public Double createModel() {
    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
    int numTrees = 10;
    int numClasses = 4;
    String featureSubsetStrategy = "auto";
    String impurity = "gini";
    int maxDepth = 9;
    int maxBins = 32;

    // create model
    RandomForestModel model = org.apache.spark.mllib.tree.RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, 12345);

    // Evaluate model on test instances and compute test error
    JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p -> new Tuple2<Double, Double>(model.predict(p.features()), p.label()));
    Double testErr = 1.0 * predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / testData.count();

    return testErr;
  }
}
