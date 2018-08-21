
using Microsoft.ML.Runtime.Api;

namespace MLTutorial1.Models
{
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }
}
