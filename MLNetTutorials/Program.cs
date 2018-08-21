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
using MLNetTutorials.Models;
using TextLoader = Microsoft.ML.Data.TextLoader;

namespace MLNetTutorials
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
