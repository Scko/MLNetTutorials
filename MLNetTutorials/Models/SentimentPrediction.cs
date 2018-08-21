
using Microsoft.ML.Runtime.Api;

namespace MLNetTutorials.Models
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Sentiment;
    }
}
