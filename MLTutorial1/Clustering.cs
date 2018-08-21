using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using MLTutorial1.Models;

namespace MLTutorial1
{
    public class Clustering
    {
        private readonly string _dataPath;
        private readonly string _modelPath;
        public Clustering()
        {
            _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data.txt");
            _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");
        }

        public PredictionModel<IrisData, ClusterPrediction> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<IrisData>(separator:','));
            pipeline.Add(new ColumnConcatenator(
                "Features",
                "SepalLength",
                "SepalWidth",
                "PetalLength",
                "PetalWidth"));
            pipeline.Add(new KMeansPlusPlusClusterer(){K = 3});
            var model = pipeline.Train<IrisData, ClusterPrediction>();
            return model;
        }

        public void Predict(PredictionModel<IrisData, ClusterPrediction> model)
        {
            IrisData Setosa = new IrisData
            {
                SepalLength = 5.1f,
                SepalWidth = 3.5f,
                PetalLength = 1.4f,
                PetalWidth = 0.2f
            };

            var prediction = model.Predict(Setosa);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
        }
    }
}
