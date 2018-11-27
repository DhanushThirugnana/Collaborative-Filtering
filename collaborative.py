from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark import SparkContext


def CF(training, testorg):
    r = 10
    noOfIter = 10
    model = ALS.train(training, r, noOfIter)

    testdata = testorg.map(lambda x: (x[0], x[1]))
    preds = model.predictAll(testdata).map(lambda x: ((x[0], x[1]), x[2]))
    joinTests = testorg.map(lambda x: ((x[0], x[1]), x[2])).join(preds)
    meanSqErr = joinTests.map(lambda x: (x[1][0] - x[1][1])**2).mean()
    print("Mean Squared Error = " + str(meanSqErr))

sc = SparkContext.getOrCreate()
file = sc.textFile("ratings.dat").map(lambda line: line.split('::'))
data = file.map(lambda y: Rating(int(y[0]), int(y[1]), float(y[2])))
(training, test) = data.randomSplit([0.6, 0.4])
training.collect()
CF(training,test)
