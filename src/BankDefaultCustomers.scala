import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.log4j.Logger
import org.apache.log4j.Level;

object BankDefaultCustomers extends App {
  //Machine Learning algorithms used are - Decision Trees, Random Forests, Naive Bayes and K-Means Clustering
  Logger.getLogger("org").setLevel(Level.ERROR);
  Logger.getLogger("akka").setLevel(Level.ERROR);
  val tempDir = "file:///home/cloudera/MLProject/spark-wh";
  val conf = new SparkConf()
    .setAppName("CardDefaulters")
    .setMaster("local[*]")
    .set("spark.executor.memory", "2g")
    .set("spark.sql.shuffle.partitions", "2")

  //Create a spark SQL session
  var spSession: SparkSession = SparkSession
    .builder()
    .appName("CardDefaulters")
    .master("local[2]")
    .config("spark.sql.warehouse.dir", tempDir)
    .getOrCreate()

  var spContext = SparkContext.getOrCreate(conf)
  

  /*--------------------------------------------------------------------------
	Load Data
	--------------------------------------------------------------------------*/
  //Load the CSV file into a RDD
  println("Loading data file :")
  val datadir = "file:///home/cloudera/MLProject/"
  val ccRDD1 = spContext.textFile(datadir + "credit-card-default-1000.csv")
  ccRDD1.cache()
  ccRDD1.take(5)
  println("Loaded lines : " + ccRDD1.count())

  //Remove the first line (contains headers) and junk lines starting with aaaaa
  val ccRDD2 = ccRDD1.filter(x =>
    !(x.startsWith("CUSTID") ||
      x.startsWith("aaaaa")))
  println("Lines after cleaning : " + ccRDD2.count())

  //Create schema to load into dataset
  import org.apache.spark.ml.linalg.{ Vector, Vectors }
  import org.apache.spark.ml.feature.LabeledPoint
  import org.apache.spark.sql.Row;
  import org.apache.spark.sql.types._

  val schema =
    StructType(
      StructField("CustId", DoubleType, false) ::
        StructField("LimitBal", DoubleType, false) ::
        StructField("Sex", DoubleType, false) ::
        StructField("Education", DoubleType, false) ::
        StructField("Marriage", DoubleType, false) ::
        StructField("Age", DoubleType, false) ::
        StructField("AvgPayDur", DoubleType, false) ::
        StructField("AvgBillAmt", DoubleType, false) ::
        StructField("AvgPayAmt", DoubleType, false) ::
        StructField("PerPaid", DoubleType, false) ::
        StructField("Defaulted", DoubleType, false) :: Nil)

  def transformToNumeric(inputStr: String): Row = {

    val attList = inputStr.split(",")

    //PR#06 - Round of age to range of 10
    val age: Double = Math.round(attList(5).toDouble / 10.0) * 10.0;

    //Normalize Sex to 1 or 2
    val sex: Double = attList(2) match {
      case "M" => 1.0
      case "F" => 0.0
      case _   => attList(2).toDouble
    }

    //Find average billed amount
    val avgBillAmt: Double = Math.abs(
      (attList(12).toDouble +
        attList(13).toDouble +
        attList(14).toDouble +
        attList(15).toDouble +
        attList(16).toDouble +
        attList(17).toDouble) / 6.0)

    //Find average pay amount
    val avgPayAmt: Double = Math.abs(
      (attList(18).toDouble +
        attList(19).toDouble +
        attList(20).toDouble +
        attList(21).toDouble +
        attList(22).toDouble +
        attList(23).toDouble) / 6.0)

    //Find average Pay duration
    val avgPayDuration: Double = Math.abs(
      (attList(6).toDouble +
        attList(7).toDouble +
        attList(8).toDouble +
        attList(9).toDouble +
        attList(10).toDouble +
        attList(11).toDouble) / 6.0)

    //Average percentage paid. add this as an additional field to see
    //if this field has any predictive capabilities. This is 
    //additional creative work that you do to see possibilities.                    
    var perPay: Double = Math.round((avgPayAmt / (avgBillAmt + 1) * 100) / 25.0) * 25.0;
    if (perPay > 100) perPay = 100

    //Filter out columns not wanted at this stage
    val values = Row(attList(0).toDouble,
      attList(1).toDouble,
      sex,
      attList(3).toDouble,
      attList(4).toDouble,
      age,
      avgPayDuration,
      avgBillAmt,
      avgPayAmt,
      perPay,
      attList(24).toDouble)
    return values
  }

  //Change to a Vector
  val ccVectors = ccRDD2.map(transformToNumeric)
  ccVectors.collect()

  println("Transformed data in Data Frame")
  val ccDf1 = spSession.createDataFrame(ccVectors, schema)
  ccDf1.printSchema()
  ccDf1.show(5)

  //Create a Dataframe for Gender
  val genderList = Array("{'sexName': 'Male', 'sexId': '1.0'}",
    "{ 'sexName':'Female','sexId':'2.0' }")
  val genderDf = spSession.read.json(spContext.parallelize(genderList))

  //Join and drop sexId
  val ccDf2 = ccDf1.join(genderDf, ccDf1("Sex") === genderDf("sexId"))
    .drop("sexId").repartition(2)

  ccDf2.show(5)

  //Add Education Name for the data Required for PR#03

  //Create a Dataframe for Education
  val eduList = Array("{'eduName': 'Graduate', 'eduId': '1.0'}",
    "{'eduName': 'University', 'eduId': '2.0'}",
    "{'eduName': 'High School', 'eduId': '3.0'}",
    "{'eduName': 'Others', 'eduId': '4.0'}")
  val eduDf = spSession.read.json(spContext.parallelize(eduList))

  //Join and drop eduId
  val ccDf3 = ccDf2.join(eduDf, ccDf1("Education") === eduDf("eduId"))
    .drop("eduId").repartition(2)

  ccDf3.show(5)

  //Create a Dataframe for Marriage
  val marriageList = Array("{'marriageName': 'Single', 'marriageId': '1.0'}",
    "{'marriageName': 'Married', 'marriageId': '2.0'}",
    "{'marriageName': 'Others', 'marriageId': '3.0'}")
  val marriageDf = spSession.read.json(spContext.parallelize(marriageList))

  //Join and drop eduId
  val ccDf4 = ccDf3.join(marriageDf, ccDf1("Marriage") === marriageDf("marriageId"))
    .drop("marriageId").repartition(2)

  println("Data frame after all enhancements :")
  ccDf4.printSchema()
  ccDf4.show(5)

  /*--------------------------------------------------------------------------
	Do analysis as required by the problem statement
	--------------------------------------------------------------------------*/
  //Create a temp view
  ccDf4.createOrReplaceTempView("CCDATA")

  //PR#02 solution
  val PR02 = spSession.sql(
    "SELECT sexName, count(*) as Total, " +
      " SUM(Defaulted) as Defaults, " +
      " ROUND(SUM(Defaulted) * 100 / count(*)) as PerDefault " +
      " FROM CCDATA GROUP BY sexName");
  println("Solution for PR#02")
  PR02.show()

  //PR#03 solution
  val PR03 = spSession.sql(
    "SELECT marriageName, eduName, count(*) as Total," +
      " SUM(Defaulted) as Defaults, " +
      " ROUND(SUM(Defaulted) * 100 / count(*)) as PerDefault " +
      " FROM CCDATA GROUP BY marriageName, eduName " +
      " ORDER BY 1,2");
  println("Solution for PR#03")
  PR03.show()

  //PR#04 solution
  val PR04 = spSession.sql(
    "SELECT AvgPayDur, count(*) as Total, " +
      " SUM(Defaulted) as Defaults, " +
      " ROUND(SUM(Defaulted) * 100 / count(*)) as PerDefault " +
      " FROM CCDATA GROUP BY AvgPayDur ORDER BY 1");
  println("Solution for PR#04")
  PR04.show()

  println("Correlation Analysis :")
  for (field <- schema.fields) {
    if (!field.dataType.equals(StringType)) {
      println("Correlation between Defaulted and " + field.name +
        " = " + ccDf4.stat.corr("Defaulted", field.name))
    }
  }
  
  /*--------------------------------------------------------------------------
	Prepare for Machine Learning
	--------------------------------------------------------------------------*/
	//Transform to a Data Frame for input to Machine Learing
	//Drop columns that are not required (low correlation / strings)
	
	def transformToLabelVectors(inStr : Row ) : LabeledPoint = { 
		//Use CustId as label. We can track the customers and their predictions
		//We will add defaulted later
	    val labelVectors = new LabeledPoint(inStr.getDouble(0) , 
									Vectors.dense(inStr.getDouble(2),
											inStr.getDouble(3),
											inStr.getDouble(4),
											inStr.getDouble(5),
											inStr.getDouble(6),
											inStr.getDouble(7),
											inStr.getDouble(8),
											inStr.getDouble(9)));
	    return labelVectors
	}
	val ccRDD3 = ccDf4.rdd.repartition(2);
	val ccLabelVectors = ccRDD3.map(transformToLabelVectors)
	ccLabelVectors.collect()
	
	val ccDf5 = spSession.createDataFrame(ccLabelVectors, classOf[LabeledPoint] )
	ccDf5.cache()
	
	//Now add Defaulted as new column
	val ccMap = ccDf4.select("CustId", "Defaulted")
	
	val ccDf6 = ccDf5.join(ccMap, ccDf5("label") === ccMap("CustId"))
						.drop("label").repartition(2)
						
	println("Data for Classification: ")
	ccDf6.show()

	//Split into training and testing data
	val Array(trainingData, testData) = ccDf6.randomSplit(Array(0.7, 0.3))
	trainingData.count()
	testData.count()
	
	/*--------------------------------------------------------------------------
	Machine Learning - Classification
	--------------------------------------------------------------------------*/
	//PR#05 Do Predictions - to predict defaults. Use multiple classification
	//algorithms to see which ones provide the best results
	
	import org.apache.spark.ml.classification.DecisionTreeClassifier
	import org.apache.spark.ml.classification.RandomForestClassifier
	import org.apache.spark.ml.classification.NaiveBayes
	import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
	
	val evaluator = new MulticlassClassificationEvaluator()
	evaluator.setPredictionCol("Prediction")
	evaluator.setLabelCol("Defaulted")
	evaluator.setMetricName("accuracy")
	
	//Do Decision Trees **********
	val dtClassifier = new DecisionTreeClassifier()
	dtClassifier.setLabelCol("Defaulted")
	dtClassifier.setPredictionCol("Prediction")
	dtClassifier.setMaxDepth(4)
	val dtModel = dtClassifier.fit(trainingData)
	val dtPredictions = dtModel.transform(testData)
	println("\nDecision Trees Accuracy = " + evaluator.evaluate(dtPredictions))
	
	//Do Random Forests **********
	val rfClassifier = new RandomForestClassifier()
	rfClassifier.setLabelCol("Defaulted")
	rfClassifier.setPredictionCol("Prediction")
	val rfModel = rfClassifier.fit(trainingData)
	val rfPredictions = rfModel.transform(testData)
	println("\nRandom Forests Accuracy = " + evaluator.evaluate(rfPredictions))
	
	//Do Naive Bayes **********
	val nbClassifier = new NaiveBayes()
	nbClassifier.setLabelCol("Defaulted")
	nbClassifier.setPredictionCol("Prediction")
	val nbModel = nbClassifier.fit(trainingData)
	val nbPredictions = nbModel.transform(testData)
	println("\nNaive Bayes Accuracy = " + evaluator.evaluate(nbPredictions))
	
	/*--------------------------------------------------------------------------
	Machine Learning - Clustering
	--------------------------------------------------------------------------*/
	//PR#06 Group data into 4 groups based on the said parameters
	
	val ccDf7 = ccDf4.select("Sex","Education","Marriage","Age","CustId")
	println("Input Data for Clustering :")
	ccDf7.show()
	
	//Perform centering and scaling
	val meanVal = ccDf7.agg(avg("Sex"), avg("Education"),avg("Marriage"),
    		avg("Age")).collectAsList().get(0)
    		
    val stdVal = ccDf7.agg(stddev("Sex"), stddev("Education"),
    		stddev("Marriage"),stddev("Age")).collectAsList().get(0)
    		
    val bcMeans=spContext.broadcast(meanVal)
	val bcStdDev=spContext.broadcast(stdVal)
	
	def centerAndScale(inRow : Row ) : LabeledPoint  = {
	    val meanArray=bcMeans.value
	    val stdArray=bcStdDev.value
	    
	    var retArray=Array[Double]()
	    
	    for (i <- 0 to inRow.size - 2)  {
	    	val csVal = ( inRow.getDouble(i) - meanArray.getDouble(i)) /
	    					 stdArray.getDouble(i)
	        retArray = retArray :+ csVal
	    }

	    return  new LabeledPoint(inRow.getDouble(4),Vectors.dense(retArray))
	} 
	
    val ccRDD4 = ccDf7.rdd.repartition(2);
	val ccRDD5 = ccRDD4.map(centerAndScale)
	ccRDD5.collect()
	 
	val ccDf8 = spSession.createDataFrame(ccRDD5, classOf[LabeledPoint] )

	println("Data ready for Clustering")
	ccDf8.select("label","features").show(10)
	
	import  org.apache.spark.ml.clustering.KMeans
	val kmeans = new KMeans()
	kmeans.setK(4)
	kmeans.setSeed(1L)
	
	//Perform K-Means Clustering
	val model = kmeans.fit(ccDf8)
	val predictions = model.transform(ccDf8)
	

	println("Groupings :")
	predictions.groupBy("prediction").count().show()
	println("Customer Assignments :")
	predictions.select("label","prediction").show()

	spContext.stop()

}