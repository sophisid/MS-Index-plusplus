package io.github.algorithms.msindexpp.ml;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PortableGBDTModel {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final List<String> featureNames;
    private final double learningRate;
    private final double initScore;
    private final List<Tree> trees;
    private final Double constantProbability;
    private final Double conformalThreshold;

    private PortableGBDTModel(List<String> featureNames, double learningRate, double initScore, List<Tree> trees,
                              Double constantProbability, Double conformalThreshold) {
        this.featureNames = featureNames;
        this.learningRate = learningRate;
        this.initScore = initScore;
        this.trees = trees;
        this.constantProbability = constantProbability;
        this.conformalThreshold = conformalThreshold;
    }

    public static PortableGBDTModel load(File modelFile) throws IOException {
        JsonNode root = MAPPER.readTree(modelFile);
        List<String> featureNames = new ArrayList<>();
        for (JsonNode featureNode : root.withArray("feature_names")) {
            featureNames.add(featureNode.asText());
        }

        List<Tree> trees = new ArrayList<>();
        for (JsonNode treeNode : root.withArray("trees")) {
            trees.add(new Tree(
                    toIntArray(treeNode.withArray("children_left")),
                    toIntArray(treeNode.withArray("children_right")),
                    toIntArray(treeNode.withArray("feature")),
                    toDoubleArray(treeNode.withArray("threshold")),
                    toDoubleArray(treeNode.withArray("value"))
            ));
        }

        return new PortableGBDTModel(
                featureNames,
                root.path("learning_rate").asDouble(),
                root.path("init_score").asDouble(),
                trees,
                root.has("constant_probability") ? root.path("constant_probability").asDouble() : null,
                root.has("conformal_threshold")   ? root.path("conformal_threshold").asDouble()   : null
        );
    }

    /** Returns the conformal threshold calibrated during training, or null for older models. */
    public Double getConformalThreshold() {
        return conformalThreshold;
    }

    public double predictProbability(TraversalFeatureSnapshot snapshot) {
        if (constantProbability != null) {
            return constantProbability;
        }

        Map<String, Double> featureMap = snapshot.toFeatureMap();
        double rawScore = initScore;
        for (Tree tree : trees) {
            rawScore += learningRate * tree.predict(featureMap, featureNames);
        }
        return 1d / (1d + Math.exp(-rawScore));
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

    private static final class Tree {
        private final int[] childrenLeft;
        private final int[] childrenRight;
        private final int[] features;
        private final double[] thresholds;
        private final double[] values;

        private Tree(int[] childrenLeft, int[] childrenRight, int[] features, double[] thresholds, double[] values) {
            this.childrenLeft = childrenLeft;
            this.childrenRight = childrenRight;
            this.features = features;
            this.thresholds = thresholds;
            this.values = values;
        }

        private double predict(Map<String, Double> featureMap, List<String> featureNames) {
            int node = 0;
            while (childrenLeft[node] != -1 && childrenRight[node] != -1) {
                int featureIndex = features[node];
                double featureValue = featureIndex >= 0 && featureIndex < featureNames.size()
                        ? featureMap.getOrDefault(featureNames.get(featureIndex), 0d)
                        : 0d;
                node = featureValue <= thresholds[node] ? childrenLeft[node] : childrenRight[node];
            }
            return values[node];
        }
    }
}
