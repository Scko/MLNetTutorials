
using Microsoft.ML.Runtime.Api;

namespace MLNetTutorials.Models
{
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
