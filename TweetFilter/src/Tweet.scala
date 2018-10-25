import scala.tools.asm.Type


class Tweet (fields: Array[String]) extends Serializable {
  
  val tweet_id: Long = fields(0).toLong
  val text: String = fields(2)
  val created_at: String = fields(3)
  
  
  
  def getTweet_id : Long = {
    return tweet_id
  }
  //getter
  def getText : String = {
    return text
  }

  

  
  def getCreated_at : String = {
    return created_at
  }
    
  override def toString: String =  tweet_id + "\t" + text + "\t" + created_at 
}