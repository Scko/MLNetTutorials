using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using MLTutorial1.Models;

namespace MLTutorial1
{
    public class BinaryClassification
    {
        private readonly string _dataPath;
        private readonly string _testDataPath;
        private readonly string _modelpath;
        public BinaryClassification()
        {
            _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-data.tsv");
            _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "wikipedia-detox-250-line-test.tsv");
            _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        }

        public async Task<PredictionModel<SentimentData, SentimentPrediction>> Train()
        {
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>());

            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));

            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 });

            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();

            await model.WriteAsync(_modelpath);

            return model;
        }

        public void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>();

            var evaluator = new BinaryClassificationEvaluator();

            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }

        public void Predict(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Please refrain from adding nonsense to Wikipedia."
                },
                new SentimentData
                {
                    SentimentText = "He is the best, and the article should say that."
                }
            };

            var predictions = model.Predict(sentiments);

            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");

            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"Sentiment: {item.sentiment.SentimentText} | Prediction: {(item.prediction.Sentiment ? "Positive" : "Negative")}");
            }
            Console.WriteLine();
        }
    }
}
