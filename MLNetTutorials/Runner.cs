using System;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using MLNetTutorials.Models;

namespace MLNetTutorials
{
    public class Runner
    {

        public void TutorialOne()
        {
            var pipeline = new LearningPipeline();

            var dataPath = @"C:\Test\MLNetTutorials\MLNetTutorials\Data\iris.data.txt";
            pipeline.Add(new TextLoader(dataPath).CreateFrom<IrisData>(separator: ','));

            pipeline.Add(new Dictionarizer("Label"));

            pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            //Learning algorithm
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            var model = pipeline.Train<IrisData, IrisPrediction>();

            var prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
        }

        public async Task BinaryTutorial()
        {
            var binaryClassification = new BinaryClassification();
            var model = await binaryClassification.Train();

            binaryClassification.Evaluate(model);

            binaryClassification.Predict(model);
        }

        public async Task RegressionTutorial()
        {
            var regression = new RegressionTaxi();
            var model = await regression.Train();

            regression.Evaluate(model);

            var trip1 = new TaxiTrip
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripDistance = 10.33f,
                PaymentType = "CSH",
                FareAmount = 0 // predict it. actual = 29.5
            };

            TaxiTripFarePrediction prediction = model.Predict(trip1);
            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.FareAmount);

        }

        public async Task ClusteringTutorial()
        {
            var clustering = new Clustering();
            var model = clustering.Train();

            clustering.Predict(model);
        }
    }
}
