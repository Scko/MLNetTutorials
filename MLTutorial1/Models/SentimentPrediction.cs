
using Microsoft.ML.Runtime.Api;

namespace MLTutorial1.Models
{
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Sentiment;
    }
}
