using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using MLTutorial1.Models;
using TextLoader = Microsoft.ML.Data.TextLoader;

namespace MLTutorial1
{
    class Program
    {
        static async Task Main(string[] args)
        {
            var runner = new Runner();

            //runner.TutorialOne();
            //await runner.BinaryTutorial();
            //await runner.RegressionTutorial();
            await runner.ClusteringTutorial();

            Console.WriteLine("Press enter to close...");
            Console.ReadLine();
        }

    }
}
