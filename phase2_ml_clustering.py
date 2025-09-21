from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def main():
    spark = SparkSession.builder \
        .appName("Phase2_ML_HighDemandClustering") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet("hdfs://namenode/output/cleaned_data/final_processed")
    print("\nCleaned data loaded.")

    feature_cols = ["PULocationID", "pickup_hour", "pickup_weekday"]
    df = df.select(*feature_cols)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df).select("features")

    kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=5, seed=42)
    model = kmeans.fit(df)
    clustered_df = model.transform(df)

    evaluator = ClusteringEvaluator(predictionCol="cluster", featuresCol="features", metricName="silhouette", distanceMeasure="squaredEuclidean")
    silhouette = evaluator.evaluate(clustered_df)

    print(f"Clustering complete. Silhouette Score: {silhouette:.2f}")

    clustered_df.write.mode("overwrite").parquet("hdfs://namenode/output/ml/HighDemandClusters")

    model.write().overwrite().save("hdfs://namenode/output/ml/models/KMeans_HighDemand_model")

    spark.stop()

if __name__ == "__main__":
    main()
