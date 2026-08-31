package io.github.algorithms.hybrid;

import io.github.io.DataManager;

import static io.github.utils.Parameters.K;
import static io.github.utils.Parameters.N;
import static io.github.utils.Parameters.qLen;
import static io.github.utils.Parameters.selectedVariatesIdx;

public class QueryDecisionFeatureExtractor {
    private final double avgSeriesLength;
    private final double stdSeriesLength;
    private final double totalSubsequences;

    public QueryDecisionFeatureExtractor() {
        final double[] lengths = new double[N];
        double subsequenceCount = 0d;
        for (int i = 0; i < N; i++) {
            lengths[i] = DataManager.getM(i);
            subsequenceCount += Math.max(0, DataManager.getM(i) - qLen + 1);
        }
        avgSeriesLength = mean(lengths);
        stdSeriesLength = std(lengths, avgSeriesLength);
        totalSubsequences = subsequenceCount;
    }

    public DecisionFeatureVector extract(double[][] query) {
        final double[] perChannelMean = new double[selectedVariatesIdx.length];
        final double[] perChannelVariance = new double[selectedVariatesIdx.length];
        final double[] perChannelAbsMean = new double[selectedVariatesIdx.length];
        final double[] perChannelStd = new double[selectedVariatesIdx.length];
        final double[] perChannelEnergy = new double[selectedVariatesIdx.length];

        for (int i = 0; i < selectedVariatesIdx.length; i++) {
            final double[] channel = query[selectedVariatesIdx[i]];
            double sum = 0d;
            double sumSquares = 0d;
            double sumAbs = 0d;
            for (double value : channel) {
                sum += value;
                sumSquares += value * value;
                sumAbs += Math.abs(value);
            }

            final double channelMean = sum / channel.length;
            final double variance = Math.max(0d, sumSquares / channel.length - channelMean * channelMean);

            perChannelMean[i] = channelMean;
            perChannelVariance[i] = variance;
            perChannelAbsMean[i] = sumAbs / channel.length;
            perChannelStd[i] = Math.sqrt(variance);
            perChannelEnergy[i] = sumSquares / channel.length;
        }

        final double queryStdMean = mean(perChannelStd);
        final double queryEnergyMean = mean(perChannelEnergy);

        return new DecisionFeatureVector(new double[]{
                K,
                avgSeriesLength,
                stdSeriesLength,
                totalSubsequences,
                avgSeriesLength <= 0d ? 0d : query[0].length / avgSeriesLength,
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
            return 0d;
        }
        double sum = 0d;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    private static double std(double[] values, double mean) {
        if (values.length == 0) {
            return 0d;
        }
        double sum = 0d;
        for (double value : values) {
            double diff = value - mean;
            sum += diff * diff;
        }
        return Math.sqrt(sum / values.length);
    }
}
