from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, isnan, isnull, avg, sum as _sum
from pyspark.sql.types import StringType, NumericType, IntegerType, TimestampType

def main():
    spark = SparkSession.builder \
        .appName("Phase2_Task2_Cleaning") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet("hdfs://namenode/input/data.parquet")
    print("\nData loaded Successfully.")

    # a) Missing Value Handling
    print("\nNull values per column:")
    null_counts_expr = []
    for c in df.columns:
        dtype = df.schema[c].dataType
        if isinstance(dtype, NumericType):
            expr_ = count(when(isnull(col(c)) | isnan(col(c)), c)).alias(c)
        else:
            expr_ = count(when(isnull(col(c)), c)).alias(c)
        null_counts_expr.append(expr_)

    null_counts = df.select(null_counts_expr)
    null_counts.show()

    numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ["double", "int", "float", "long"]]
    for col_name in numeric_cols:
        median_val = df.approxQuantile(col_name, [0.5], 0.01)[0]
        if median_val is not None:
            df = df.fillna({col_name: median_val})

    categorical_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    for col_name in categorical_cols:
        mode_val = df.groupBy(col_name).count().orderBy("count", ascending=False).first()
        if mode_val:
            df = df.fillna({col_name: mode_val[0]})

    threshold = 0.5 * df.count()
    for c in df.columns:
        dtype = df.schema[c].dataType
        if isinstance(dtype, NumericType):
            null_count = df.filter(isnull(col(c)) | isnan(col(c))).count()
        else:
            null_count = df.filter(isnull(col(c))).count()
        if null_count > threshold:
            print(f"Dropping column: {c} (too many nulls: {null_count})")
            df = df.drop(c)

    # c) converting column data types
    print("\nNow converting column data types")
    conversions = {
        "tpep_pickup_datetime": TimestampType(),
        "tpep_dropoff_datetime": TimestampType(),
        "pickup_hour": IntegerType(),
        "pickup_day": IntegerType(),
        "pickup_weekday": IntegerType(),
        "pickup_month": IntegerType()
    }
    for col_name, new_type in conversions.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast(new_type))
            print(f"Converted {col_name} to {new_type.simpleString()}")

    # b) Removing Outliers
    print("\nRemoving Outliers:")
    total_before = df.count()
    for col_name in numeric_cols:
        q1, q3 = df.approxQuantile(col_name, [0.25, 0.75], 0.05)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df.filter((col(col_name) >= lower_bound) & (col(col_name) <= upper_bound))
        print(f"{col_name}: kept rows between {lower_bound:.2f} and {upper_bound:.2f}")

    total_after = df.count()
    print(f"\nRemoved {total_before - total_after} outlier rows. Remaining rows: {total_after}")

    # d) Aggregation and Grouping Summary
    print("\nAggregation and Grouping Summary:")
    if 'pickup_hour' in df.columns and 'fare_amount' in df.columns:
        df.groupBy("pickup_hour").agg(avg("fare_amount").alias("avg_fare")).orderBy("pickup_hour").show()

    if 'pickup_weekday' in df.columns and 'trip_distance' in df.columns:
        df.groupBy("pickup_weekday").agg(avg("trip_distance").alias("avg_distance")).orderBy("pickup_weekday").show()

    if 'payment_type' in df.columns and 'tip_amount' in df.columns:
        df.groupBy("payment_type").agg(_sum("tip_amount").alias("total_tips")).orderBy("payment_type").show()

    # e) Writting Processed Data
    df.write.mode("overwrite").parquet("hdfs://namenode/output/cleaned_data/final_processed")
    print("\nFinal cleaned data saved to /cleaned_data/final_processed")

    spark.stop()

if __name__ == "__main__":
    main()