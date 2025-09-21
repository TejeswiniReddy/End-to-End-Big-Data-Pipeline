from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


def main():
    try:
        active_spark = SparkSession.getActiveSession()
        if active_spark is not None:
            print("Stopping previous active SparkSession.")
            active_spark.stop()
    except Exception as e:
        print(f"No active session found or error checking it: {e}")

    spark = SparkSession.builder \
        .appName("Phase2_Task3_FarePrediction") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet("hdfs://namenode/output/cleaned_data/final_processed")
    print("\nCleaned data loaded.")

    feature_cols = [
        "trip_distance", "pickup_hour", "pickup_weekday",
        "pickup_month", "passenger_count", "pickup_day"
    ]
    df = df.select(*feature_cols, "fare_amount")

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df).select("features", "fare_amount")

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    models = {
        "LinearRegression": LinearRegression(featuresCol="features", labelCol="fare_amount"),
        "RandomForest": RandomForestRegressor(featuresCol="features", labelCol="fare_amount")
    }

    evaluator = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")

    for name, model in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(stages=[model])
        model_fit = pipeline.fit(train_df)
        predictions = model_fit.transform(test_df)

        rmse = evaluator.evaluate(predictions)
        r2 = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2").evaluate(predictions)

        print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.2f}")

        predictions.select("features", "fare_amount", "prediction") \
            .write.mode("overwrite") \
            .parquet(f"hdfs://namenode/output/ml/{name}_fare_predictions")

        model_fit.write().overwrite().save(f"hdfs://namenode/output/ml/models/{name}_fare_model")

    print("\nTuning and training GBTRegressor...")
    gbt = GBTRegressor(featuresCol="features", labelCol="fare_amount")

    paramGrid = ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [5, 7, 10]) \
        .addGrid(gbt.maxIter, [50, 100]) \
        .build()

    crossval = CrossValidator(estimator=gbt,
                               estimatorParamMaps=paramGrid,
                               evaluator=evaluator,
                               numFolds=3)

    gbt_model = crossval.fit(train_df)
    gbt_predictions = gbt_model.transform(test_df)

    gbt_rmse = evaluator.evaluate(gbt_predictions)
    gbt_r2 = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2").evaluate(gbt_predictions)

    print(f"Tuned GBTRegressor - RMSE: {gbt_rmse:.2f}, R²: {gbt_r2:.2f}")

    gbt_predictions.select("features", "fare_amount", "prediction") \
        .write.mode("overwrite") \
        .parquet("hdfs://namenode/output/ml/TunedGBTRegressor_fare_predictions")

    gbt_model.write().overwrite().save("hdfs://namenode/output/ml/models/TunedGBTRegressor_fare_model")

    spark.stop()

if __name__ == "__main__":
    main()
