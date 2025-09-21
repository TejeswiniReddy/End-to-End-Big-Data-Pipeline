from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, when


def main():
    spark = SparkSession.builder \
        .appName("Phase2_Task2_Analysis_Objectives") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet("hdfs://namenode/output/cleaned_data/final_processed")
    print("\nCleaned data loaded.")

    trip_distance_patterns = df.groupBy("trip_distance").count().orderBy("trip_distance")
    trip_distance_patterns.write.mode("overwrite").csv("hdfs://namenode/output/analysis/trip_distance_patterns")

    df = df.withColumn("cost_per_mile", when(col("trip_distance") > 0, col("total_amount") / col("trip_distance")).otherwise(None))
    duration_vs_cost = df.select("trip_duration", "cost_per_mile")
    duration_vs_cost.write.mode("overwrite").csv("hdfs://namenode/output/analysis/duration_vs_cost")

    peak_hours = df.groupBy("pickup_hour").count().orderBy("pickup_hour")
    peak_hours.write.mode("overwrite").csv("hdfs://namenode/output/analysis/peak_hours")

    df = df.withColumn("tip_percentage", when(col("fare_amount") > 0, (col("tip_amount") / col("fare_amount")) * 100).otherwise(0))
    tipping_behavior = df.groupBy("payment_type").agg(avg("tip_percentage").alias("avg_tip_percentage"))
    tipping_behavior.write.mode("overwrite").csv("hdfs://namenode/output/analysis/tipping_behavior")

    weekday_trends = df.groupBy("pickup_weekday").count().orderBy("pickup_weekday")
    weekday_trends.write.mode("overwrite").csv("hdfs://namenode/output/analysis/weekday_trends")

    print("\nAll analysis objectives completed and saved.")

    spark.stop()

if __name__ == "__main__":
    main()