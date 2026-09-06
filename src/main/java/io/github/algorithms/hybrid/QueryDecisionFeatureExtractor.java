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
        final double[] perChannelRange = new double[selectedVariatesIdx.length];
        final double[] perChannelDiffEnergy = new double[selectedVariatesIdx.length];

        for (int i = 0; i < selectedVariatesIdx.length; i++) {
            final double[] channel = query[selectedVariatesIdx[i]];
            double sum = 0d;
            double sumSquares = 0d;
            double sumAbs = 0d;
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            double sumSquaredDiffs = 0d;
            double previous = 0d;
            for (int j = 0; j < channel.length; j++) {
                final double value = channel[j];
                sum += value;
                sumSquares += value * value;
                sumAbs += Math.abs(value);
                min = Math.min(min, value);
                max = Math.max(max, value);
                if (j > 0) {
                    final double difference = value - previous;
                    sumSquaredDiffs += difference * difference;
                }
                previous = value;
            }

            final double channelMean = sum / channel.length;
            final double variance = Math.max(0d, sumSquares / channel.length - channelMean * channelMean);

            perChannelMean[i] = channelMean;
            perChannelVariance[i] = variance;
            perChannelAbsMean[i] = sumAbs / channel.length;
            perChannelStd[i] = Math.sqrt(variance);
            perChannelEnergy[i] = sumSquares / channel.length;
            perChannelRange[i] = max - min;
            perChannelDiffEnergy[i] = channel.length <= 1 ? 0d : sumSquaredDiffs / (channel.length - 1);
        }

        final double queryStdMean = mean(perChannelStd);
        final double queryEnergyMean = mean(perChannelEnergy);
        final double queryRangeMean = mean(perChannelRange);
        final double queryDiffEnergyMean = mean(perChannelDiffEnergy);

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
                std(perChannelEnergy, queryEnergyMean),
                queryRangeMean,
                std(perChannelRange, queryRangeMean),
                queryDiffEnergyMean,
                std(perChannelDiffEnergy, queryDiffEnergyMean)
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
