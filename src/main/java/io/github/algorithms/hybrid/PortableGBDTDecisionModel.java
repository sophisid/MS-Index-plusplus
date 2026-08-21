package io.github.algorithms.hybrid;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.algorithms.AlgorithmType;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PortableGBDTDecisionModel {
    private final String[] featureNames;
    private final double learningRate;
    private final double initialScore;
    private final Double constantProbability;
    private final List<RegressionTree> trees;

    private PortableGBDTDecisionModel(
            String[] featureNames,
            double learningRate,
            double initialScore,
            Double constantProbability,
            List<RegressionTree> trees
    ) {
        this.featureNames = featureNames;
        this.learningRate = learningRate;
        this.initialScore = initialScore;
        this.constantProbability = constantProbability;
        this.trees = trees;
    }

    public static PortableGBDTDecisionModel load(Path path) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(path.toFile());
            String[] featureNames = mapper.convertValue(root.get("feature_names"), String[].class);

            if (!Arrays.equals(featureNames, DecisionFeatureVector.FEATURE_NAMES)) {
                throw new IllegalArgumentException("Feature names in model do not match Java feature extractor");
            }

            List<RegressionTree> trees = new ArrayList<>();
            JsonNode treeNodes = root.get("trees");
            if (treeNodes != null) {
                for (JsonNode treeNode : treeNodes) {
                    trees.add(new RegressionTree(
                            toIntArray(treeNode.get("children_left")),
                            toIntArray(treeNode.get("children_right")),
                            toIntArray(treeNode.get("feature")),
                            toDoubleArray(treeNode.get("threshold")),
                            toDoubleArray(treeNode.get("value"))
                    ));
                }
            }

            Double constantProbability = root.has("constant_probability")
                    ? root.get("constant_probability").asDouble()
                    : null;

            return new PortableGBDTDecisionModel(
                    featureNames,
                    root.path("learning_rate").asDouble(),
                    root.path("init_score").asDouble(),
                    constantProbability,
                    trees
            );
        } catch (IOException e) {
            throw new RuntimeException("Failed to load hybrid model from " + path, e);
        }
    }

    public AlgorithmType predict(DecisionFeatureVector featureVector) {
        return predictProbability(featureVector) >= 0.5 ? AlgorithmType.MASS : AlgorithmType.MSINDEX;
    }

    public double predictProbability(DecisionFeatureVector featureVector) {
        double[] features = featureVector.values();
        if (features.length != featureNames.length) {
            throw new IllegalArgumentException("Expected " + featureNames.length + " features but got " + features.length);
        }
        if (constantProbability != null) {
            return constantProbability;
        }

        double rawScore = initialScore;
        for (RegressionTree tree : trees) {
            rawScore += learningRate * tree.predict(features);
        }
        return 1.0 / (1.0 + Math.exp(-rawScore));
    }

    private static int[] toIntArray(JsonNode node) {
        int[] values = new int[node.size()];
        for (int i = 0; i < node.size(); i++) {
            values[i] = node.get(i).asInt();
        }
        return values;
    }

    private static double[] toDoubleArray(JsonNode node) {
        double[] values = new double[node.size()];
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
                int featureIndex = features[node];
                node = sample[featureIndex] <= thresholds[node] ? childrenLeft[node] : childrenRight[node];
            }
            return values[node];
        }
    }
}
