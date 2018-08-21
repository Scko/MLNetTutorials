using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Api;

namespace MLTutorial1.Models
{
    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
}
