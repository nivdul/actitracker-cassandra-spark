package com.actitracker.model;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

public class MulticlassLogisticRegression {

  JavaRDD<LabeledPoint> trainingData;
  JavaRDD<LabeledPoint> testData;

  public MulticlassLogisticRegression(JavaRDD<LabeledPoint> trainingData, JavaRDD<LabeledPoint> testData) {
    this.trainingData = trainingData;
    this.testData = testData;
  }

  public Double createmodel() {

    LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
        .setNumClasses(4)
        .run(trainingData.rdd());

    JavaRDD<Tuple2<Object, Object>> predictionAndLabel = testData.map(p -> new Tuple2<>(model.predict(p.features()), p.label()));

    // Evaluate metrics
    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabel.rdd());
    Double precision = metrics.precision();

    return precision;
  }

}
