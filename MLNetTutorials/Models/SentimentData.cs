﻿
using Microsoft.ML.Runtime.Api;

namespace MLNetTutorials.Models
{
    public class SentimentData
    {
        [Column(ordinal: "0", name: "Label")]
        public float Sentiment;
        [Column(ordinal: "1")]
        public string SentimentText;
    }
}