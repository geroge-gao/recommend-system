from pyspark import SparkConf, SparkContext
# 创建SparkConf和SparkContext
conf = SparkConf().setMaster("local").setAppName("lichao-wordcount")
sc = SparkContext(conf=conf)
# 输入的数据
data = ["hello", "world", "hello", "word", "count", "count", "hello"]
# 将Collection的data转化为spark中的rdd并进行操作
rdd = sc.parallelize(data)
resultRdd = rdd.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
# rdd转为collecton并打印
resultColl = resultRdd.collect()
for line in resultColl:
    print(line)