import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scala.util.matching.Regex
import util.control.Breaks._



object Filter {
  def main(args: Array[String]): Unit = {
    
    // executing Spark on your machine, using 6 threads
    val conf = new SparkConf().setMaster("local[*]").setAppName("Filter")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    
    
    println("***********************************************")
    val t0 = System.nanoTime()   // Timer to compute the execution time (start time)
    
    val tweetRDD: RDD[String] =  sc.textFile("tweets15M.txt")
//    println("Number of Tweets:  " + tweetRDD.count())
    
    //**********************************************************************
    //********************************************************************
    //******************* Create english tweets (class tweet) ****************
    
    
    def stringtoTweet (s: String) : Tweet =  // convert a string to a tweet ( id, text, created_at)
    {
      val array : Array[String] = s.split("\t")
      val tweet = new Tweet(array)
      return tweet
    }

    val englishRDD : RDD[Tweet] = tweetRDD.map(f=>stringtoTweet(f))
   // englishRDD.take(3).foreach(println)
//    println("Number of English Tweets:  " + englishRDD.count())

    
    //**************************************************************************
    //********************************************************************************
    //***********************Remove URL from the tweets **********************************
    
    val pattern = new Regex("^" +"(?:(?:https?|ftp)://)" +"(?:\\S+(?::\\S*)?@)?" +
        "(?:" +"(?!(?:10|127)(?:\\.\\d{1,3}){3})" +"(?!(?:169\\.254|192\\.168)(?:\\.\\d{1,3}){2})" +
      "(?!172\\.(?:1[6-9]|2\\d|3[0-1])(?:\\.\\d{1,3}){2})" +"(?:[1-9]\\d?|1\\d\\d|2[01]\\d|22[0-3])" +
      "(?:\\.(?:1?\\d{1,2}|2[0-4]\\d|25[0-5])){2}" +"(?:\\.(?:[1-9]\\d?|1\\d\\d|2[0-4]\\d|25[0-4]))" +"|" +
      "(?:(?:[a-z\\u00a1-\\uffff0-9]-*)*[a-z\\u00a1-\\uffff0-9]+)" +
      "(?:\\.(?:[a-z\\u00a1-\\uffff0-9]-*)*[a-z\\u00a1-\\uffff0-9]+)*" +
      "(?:\\.(?:[a-z\\u00a1-\\uffff]{2,}))" +"\\.?" +
       ")" +"(?::\\d{2,5})?" +"(?:[/?#]\\S*)?" +"$", "i");
    
      
      def detectURL ( str: String) : Boolean =  // return true if the tweet contain a valid url 
      {
         
        val array : Array[String] = str.split(" ")

        breakable
        {
          for (s <- array) 
           {
              pattern.findFirstMatchIn(s) match 
                      {
                      case Some(_) => return true;break;
                      case None => 
                     }
                        }
           }
    
         return false
      }
      
      val tweetWoURLRDD : RDD[Tweet] = englishRDD.filter(f => !detectURL(f.getText)) 
      //tweetWoURLRDD.take(10).foreach(println)
      //println(tweetWoURLRDD.count())
    
      //**************************************************************************
    //********************************************************************************
    //***********************Remove Usernames and hashtags from the tweets **********************************
      
     val patternHT = new Regex("(?:\\s|\\A)[##]+([A-Za-z0-9-_]+)");  //for hashTags
     val patternUN = new Regex("(?:\\s|\\A)[@]+([A-Za-z0-9-_]+)");    // for User Names
    
     def removeHTUN (s : String) : String = {    // remove hashtags and userNames from a string 
      
       return s.replaceAll(patternUN.toString(), " ").replaceAll(patternHT.toString(), " ");
     }
     
     val tweetCleanedRDD : RDD[Tweet] = tweetWoURLRDD.map(f=> new Tweet(Array(f.tweet_id.toString(),"",removeHTUN(f.text),f.created_at)))   
     // ------------------------------- above is a  new constructeur similar to the one declared in the Tweet class
     //tweetCleanedRDD.take(10).foreach(println)
     //println(tweetCleanedRDD.count()) 
    
         //**************************************************************************
    //********************************************************************************
    //***********************Use the keywords files to filter the tweetcleanedRDD **********************************
     
     val keywordsArray : Array[String] = sc.textFile("DataSets").repartition(1).map(f=>f.toLowerCase()).collect()
    
     def concat ( words : Array[String], start: Int, end: Int) :String = {
       var sb : StringBuilder = new StringBuilder();
       
       for( i <- start until  end){
           sb.append( (if (i > start) " " else "") + words(i) ) 
       }

       return sb.toString
     }

 
     def ngrams (n : Int, str: String): List[String] = {
        
       var ngrams : List[String] = List()
       var words : Array[String] =  str.split("[\\s+\\-*/\\^:;\\[\\]\\\\()_#@$%&|<>\'\",.?{}!=`~]+")
       
       for( i <- 0 until  words.length - n + 1){
         ngrams = ngrams.::(concat(words, i, i+n))
       }
       return ngrams 
        
     }

       
     def filterOnKeywords (s : String) : Boolean = {
 
        var decision : Boolean = false
        var comp_ngram : Int = 1    // compteur to choose the n of the ngram
        
        while(decision == false && comp_ngram<=6){   // 6 in the max length of my keywords (ti gain in time execution)
          
          var tokenizer : List[String] = List()      
          var i_ngram = ngrams(comp_ngram, s)
         
          tokenizer = i_ngram                    //compute the i_ngram and affect it to tokenizer to verify with the keywords
         
          val length = tokenizer.length
          var i : Int = 0
         
           while(decision == false && i<length ){
           
             if (keywordsArray contains (tokenizer(i).toLowerCase())){
                   decision = true
                 }
        
             i += 1;              
            
           }
            comp_ngram += 1;
         }
         return decision
     }

     
    
     
     val filtredTweetsRDD : RDD[Tweet] = tweetCleanedRDD.filter(f => filterOnKeywords(f.text))
     filtredTweetsRDD.coalesce(1, true).saveAsTextFile("tweetFiltred3.txt")
     
     
     
      val t1 = System.nanoTime()   // Timer to compute the execution time (end time)
      println("Total Time", t1 - t0)
    
     // terminate spark context
    sc.stop()
  
    println("done")
  }
}