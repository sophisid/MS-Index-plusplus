package io.github.algorithms.msindexpp.decision;

import io.github.io.DataManager;

import static io.github.utils.Parameters.N;
import static io.github.utils.Parameters.K;
import static io.github.utils.Parameters.channels;
import static io.github.utils.Parameters.nQueryChannels;
import static io.github.utils.Parameters.nSubsequences;
import static io.github.utils.Parameters.selectedVariatesIdx;

public class QueryDecisionFeatureExtractor {
    private final double avgSeriesLength;
    private final double stdSeriesLength;
    private final double avgSubsequencesPerSeries;

    public QueryDecisionFeatureExtractor() {
        final double[] lengths = new double[N];
        for (int i = 0; i < N; i++) {
            lengths[i] = DataManager.getM(i);
        }
        avgSeriesLength = mean(lengths);
        stdSeriesLength = std(lengths, avgSeriesLength);
        avgSubsequencesPerSeries = N == 0 ? 0 : nSubsequences / (double) N;
    }

    public DecisionFeatureVector extract(double[][] query) {
        final double[] perChannelMean = new double[selectedVariatesIdx.length];
        final double[] perChannelVariance = new double[selectedVariatesIdx.length];
        final double[] perChannelAbsMean = new double[selectedVariatesIdx.length];
        final double[] perChannelStd = new double[selectedVariatesIdx.length];
        final double[] perChannelEnergy = new double[selectedVariatesIdx.length];

        for (int i = 0; i < selectedVariatesIdx.length; i++) {
            final double[] channel = query[selectedVariatesIdx[i]];
            double sum = 0;
            double sumSquares = 0;
            double sumAbs = 0;
            for (double value : channel) {
                sum += value;
                sumSquares += value * value;
                sumAbs += Math.abs(value);
            }

            final double channelMean = sum / channel.length;
            final double variance = Math.max(0, sumSquares / channel.length - channelMean * channelMean);

            perChannelMean[i] = channelMean;
            perChannelVariance[i] = variance;
            perChannelAbsMean[i] = sumAbs / channel.length;
            perChannelStd[i] = Math.sqrt(variance);
            perChannelEnergy[i] = sumSquares / channel.length;
        }

        final double queryStdMean = mean(perChannelStd);
        final double queryEnergyMean = mean(perChannelEnergy);
        final double queryLengthRatio = avgSeriesLength == 0 ? 0 : query[0].length / avgSeriesLength;
        
        return new DecisionFeatureVector(new double[]{
                K,
                channels,
                nQueryChannels,
                N,
                avgSeriesLength,
                stdSeriesLength,
                avgSubsequencesPerSeries,
                nSubsequences,
                query[0].length,
                queryLengthRatio,
                mean(perChannelMean),
                mean(perChannelVariance),
                mean(perChannelAbsMean),
                queryStdMean,
                std(perChannelStd, queryStdMean),
                queryEnergyMean,
                std(perChannelEnergy, queryEnergyMean)
        });
    }

    private static double mean(double[] values) {
        if (values.length == 0) {
            return 0;
        }
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    private static double std(double[] values, double mean) {
        if (values.length == 0) {
            return 0;
        }
        double sum = 0;
        for (double value : values) {
            final double diff = value - mean;
            sum += diff * diff;
        }
        return Math.sqrt(sum / values.length);
    }
}
