using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using MLNetTutorials.Models;

namespace MLNetTutorials
{
    public class RegressionTaxi
    {
        private readonly string _datapath;
        private readonly string _testdatapath;
        private readonly string _modelpath;
        public RegressionTaxi()
        {
            _datapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
            _testdatapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
            _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        }

        public async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','),
                new ColumnCopier(("FareAmount", "Label")),
                new CategoricalOneHotVectorizer("VendorId", "RateCode", "PaymentType"),
                new ColumnConcatenator("Features", "VendorId", "RateCode", "PassengerCount", "TripDistance",
                    "PaymentType"),
                new FastTreeRegressor()
            };

            var model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();

            await model.WriteAsync(_modelpath);
            return model;
        }

        public void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine($"Rms = {metrics.Rms}");
            Console.WriteLine($"RSquared = {metrics.RSquared}");
        }

    }
}
