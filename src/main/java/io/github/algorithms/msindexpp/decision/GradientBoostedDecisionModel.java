package io.github.algorithms.msindexpp.decision;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.algorithms.AlgorithmType;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class GradientBoostedDecisionModel {
    private final String[] featureNames;
    private final double learningRate;
    private final double initialScore;
    private final List<RegressionTree> trees;

    private GradientBoostedDecisionModel(
            String[] featureNames,
            double learningRate,
            double initialScore,
            List<RegressionTree> trees
    ) {
        this.featureNames = featureNames;
        this.learningRate = learningRate;
        this.initialScore = initialScore;
        this.trees = trees;
    }

    public static GradientBoostedDecisionModel load(Path path) {
        try {
            final ObjectMapper mapper = new ObjectMapper();
            final JsonNode root = mapper.readTree(path.toFile());
            final String[] featureNames = mapper.convertValue(root.get("feature_names"), String[].class);

            if (!Arrays.equals(featureNames, DecisionFeatureVector.FEATURE_NAMES)) {
                throw new IllegalArgumentException("Feature names in model do not match Java feature extractor");
            }

            final List<RegressionTree> trees = new ArrayList<>();
            for (JsonNode treeNode : root.get("trees")) {
                trees.add(new RegressionTree(
                        toIntArray(treeNode.get("children_left")),
                        toIntArray(treeNode.get("children_right")),
                        toIntArray(treeNode.get("feature")),
                        toDoubleArray(treeNode.get("threshold")),
                        toDoubleArray(treeNode.get("value"))
                ));
            }

            return new GradientBoostedDecisionModel(
                    featureNames,
                    root.get("learning_rate").asDouble(),
                    root.get("initial_score").asDouble(),
                    trees
            );
        } catch (IOException e) {
            throw new RuntimeException("Failed to load model from " + path, e);
        }
    }

    public AlgorithmType predict(DecisionFeatureVector featureVector) {
        final double[] features = featureVector.values();
        if (features.length != featureNames.length) {
            throw new IllegalArgumentException("Expected " + featureNames.length + " features but got " + features.length);
        }

        double rawScore = initialScore;
        for (RegressionTree tree : trees) {
            rawScore += learningRate * tree.predict(features);
        }
        return rawScore > 0 ? AlgorithmType.MASS : AlgorithmType.MSINDEX;
    }

    private static int[] toIntArray(JsonNode node) {
        final int[] values = new int[node.size()];
        for (int i = 0; i < node.size(); i++) {
            values[i] = node.get(i).asInt();
        }
        return values;
    }

    private static double[] toDoubleArray(JsonNode node) {
        final double[] values = new double[node.size()];
        for (int i = 0; i < node.size(); i++) {
            values[i] = node.get(i).asDouble();
        }
        return values;
    }

    private static final class RegressionTree {
        private final int[] childrenLeft;
        private final int[] childrenRight;
        private final int[] features;
        private final double[] thresholds;
        private final double[] values;

        private RegressionTree(
                int[] childrenLeft,
                int[] childrenRight,
                int[] features,
                double[] thresholds,
                double[] values
        ) {
            this.childrenLeft = childrenLeft;
            this.childrenRight = childrenRight;
            this.features = features;
            this.thresholds = thresholds;
            this.values = values;
        }

        private double predict(double[] sample) {
            int node = 0;
            while (childrenLeft[node] != -1) {
                final int featureIndex = features[node];
                node = sample[featureIndex] <= thresholds[node] ? childrenLeft[node] : childrenRight[node];
            }
            return values[node];
        }
    }
}
