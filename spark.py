from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, month, count, mean, stddev

def main():
    spark = SparkSession.builder \
        .appName("Phase2_Task1_EDA") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    input_path = "hdfs://namenode/input/data.parquet"
    df = spark.read.parquet(input_path)
    print("Data loaded from HDFS!!!")

    print("\n Schema:")
    df.printSchema()

    print("\n Sample Data:")
    df.show(5)

    print("\n Summary statistics:")
    df.describe().show()

    if 'tpep_pickup_datetime' in df.columns:
        df = df.withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))
        pickup_by_hour = df.groupBy("pickup_hour").count().orderBy("pickup_hour")
        print("\n Trips by Pickup Hour:")
        pickup_by_hour.show()

        pickup_by_hour.write.mode("overwrite").csv("hdfs://namenode/output/eda/pickup_by_hour")

    if 'trip_distance' in df.columns:
        df.select("trip_distance").describe().show()

        dist_bins = df.select("trip_distance").rdd.flatMap(lambda x: x).histogram([0,1,2,5,10,20,30,50])
        print("\n Histogram of Trip Distances:")
        print(dist_bins)

    if 'tpep_pickup_datetime' in df.columns:
        df = df.withColumn("pickup_weekday", dayofweek(col("tpep_pickup_datetime")))
        weekday_counts = df.groupBy("pickup_weekday").count().orderBy("pickup_weekday")
        print("\n Trips by Weekday:")
        weekday_counts.show()

        weekday_counts.write.mode("overwrite").csv("hdfs://namenode/output/eda/weekday_counts")

    if {'tip_amount', 'fare_amount', 'payment_type'}.issubset(df.columns):
        df = df.withColumn("tip_percent", (col("tip_amount") / col("fare_amount")) * 100)
        tip_stats = df.groupBy("payment_type").agg(mean("tip_percent").alias("avg_tip_pct"))
        print("\n Average Tip Percent by Payment Type:")
        tip_stats.show()

        tip_stats.write.mode("overwrite").csv("hdfs://namenode/output/eda/tip_percent")

    spark.stop()
    print("EDA complete.")


if __name__ == "__main__":
    main()
